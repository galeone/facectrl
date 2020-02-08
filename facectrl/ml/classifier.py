# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Package containing the tools to use during the inference."""

import logging
from enum import Enum, auto
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

from facectrl.ml.model import CVAE


class ClassificationResult(Enum):
    """A possible classification result."""

    HEADPHONES_ON = auto()
    HEADPHONES_OFF = auto()
    UNKNOWN = auto()


class Thresholds:
    """The thresholds learned by the cvae."""

    def __init__(self, on: Dict, off: Dict):
        """The thresholds to use,
        in the format {"mean": value, "variance: value}."""
        keys = {"mean", "variance"}
        if keys != set(on.keys()) or keys != set(off.keys()):
            raise ValueError("Wrong threshold format, exepected mean,variance keys.")
        self._on = on
        self._off = off

    @property
    def difference(self):
        """Absolute distance between positive and negative mean values."""
        return abs(self.on["mean"] - self.off["mean"])

    @property
    def on(self):
        return self._on

    @property
    def off(self):
        return self._off

    def asdict(self) -> Dict:
        return {
            "positive_threshold": self._on["mean"].item(),
            "positive_variance": self._on["variance"].item(),
            "negative_threshold": self._off["mean"].item(),
            "negative_variance": self._off["variance"].item(),
        }


class Classifier:
    """The Classifier object implements all the logic to trigger a detection.
    Args:
        cvae (tf.keras.Model): the previously trained cvae.
        thresholds (Thresholds): the dictionary containing the learned thresholds
                           (model selection result).
        debug (bool): when True, it enables the opencv visualization.
    """

    def __init__(self, cvae: CVAE, thresholds: Thresholds, debug: bool = False) -> None:
        self._cvae = cvae
        self._thresholds = thresholds
        # mse that keeps the batch size
        self._mse = lambda a, b: tf.math.reduce_mean(
            tf.math.squared_difference(a, b), axis=[1, 2, 3]
        )
        self._debug = debug

    @staticmethod
    def preprocess(crop: np.array) -> tf.Tensor:
        """The preprocessing operation to apply, before feeding the image to the classifier.
        Args:
            crop (np.array): a BGR image. np.uin8
        Return:
            face (tf.Tensor): the post-processed face, tf.float32, 64x64. RGB.
        """
        crop = tf.convert_to_tensor(crop)
        rgb = tf.reverse(crop, axis=[-1])  # 0,255
        # Convert to [0,1]
        rgb = tf.image.convert_image_dtype(rgb, tf.float32)
        # Convert to [-1,1]
        rgb = (rgb - 0.5) * 2.0
        rgb = tf.image.resize(rgb, (64, 64))
        if tf.equal(tf.rank(rgb), 3):
            return tf.expand_dims(rgb, axis=[0])
        return rgb

    @property
    def cvae(self) -> CVAE:
        """Get the cvae currently in use."""
        return self._cvae

    @property
    def thresholds(self) -> Thresholds:
        """Get the thresholds currently in use."""
        return self._thresholds

    @staticmethod
    def normalize(image: tf.Tensor) -> tf.Tensor:
        """Given image in [-1,1] returns image in [0,1]."""
        return (image + 1.0) / 2.0

    def __call__(self, face: tf.Tensor) -> ClassificationResult:
        """Using the cvae and the thresholds, do the classifcation of the face.
        Args:
            face (tf.Tensor): the cropped tensor (use Classifier.preprocess)
        Return:
            ClassificationResult: the result of the classification.
        """

        # if needed, ad batch size
        if tf.equal(tf.rank(face), 3):
            face = tf.expand_dims(face, axis=[0])

        classified = np.array(
            [ClassificationResult.UNKNOWN] * tf.shape(face)[0].numpy()
        )
        reconstruction = self._cvae.reconstruct(
            face
        )  # face and reconstructions have values in [-1,1]
        mse = self._mse(face, reconstruction).numpy()

        on_sigma = self._thresholds.on["variance"]
        off_sigma = self._thresholds.off["variance"]

        classified[
            np.logical_and(
                mse >= (self._thresholds.on["mean"] - 3 * on_sigma),
                mse <= (self._thresholds.on["mean"] + 3 * on_sigma),
            )
        ] = ClassificationResult.HEADPHONES_ON
        classified[
            np.logical_and(
                mse >= (self._thresholds.off["mean"] - 3 * off_sigma),
                mse <= (self._thresholds.off["mean"] + 3 * off_sigma),
            )
        ] = ClassificationResult.HEADPHONES_OFF

        for idx, element in enumerate(classified):
            if element != ClassificationResult.UNKNOWN:
                logging.info("Classified as: %s with mse %f", element, mse[idx])
                if self._debug:
                    # tf.reverse to go from RGB to BGR
                    cv2.imshow(
                        "reconstruction",
                        tf.squeeze(
                            tf.image.convert_image_dtype(
                                tf.reverse(
                                    self.normalize(reconstruction[idx]), axis=[-1]
                                ),
                                tf.uint8,
                            )
                        ).numpy(),
                    )
                    cv2.imshow(
                        "input",
                        tf.squeeze(
                            tf.image.convert_image_dtype(
                                tf.reverse(self.normalize(face[idx]), axis=[-1]),
                                tf.uint8,
                            )
                        ).numpy(),
                    )
            else:
                logging.info(
                    "Unable to classify the input. mse %s is outside of positive %f and negative %f",
                    mse,
                    self._thresholds.on["mean"],
                    self._thresholds.off["mean"],
                )

        return classified

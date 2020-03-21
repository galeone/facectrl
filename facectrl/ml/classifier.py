# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Package containing the tools to use during the inference."""

import logging
from enum import Enum
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf


class ClassificationResult(Enum):
    """A possible classification result."""

    HEADPHONES_ON = 1
    HEADPHONES_OFF = 0
    UNKNOWN = -1


class Thresholds:
    """The thresholds learned by the model."""

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
        """On dictionary."""
        return self._on

    @property
    def off(self):
        """Off dictionary."""
        return self._off

    def asdict(self) -> Dict:
        """Convert the thredshols to a dict."""
        return {
            "positive_threshold": self._on["mean"].item(),
            "positive_variance": self._on["variance"].item(),
            "negative_threshold": self._off["mean"].item(),
            "negative_variance": self._off["variance"].item(),
        }


class Classifier:
    """The Classifier object implements all the logic to trigger a detection.
    Args:
        model (tf.keras.Model): the previously trained model.
        thresholds (Thresholds): the dictionary containing the learned thresholds
                           (model selection result).
        debug (bool): when True, it enables the opencv visualization.
    """

    def __init__(
        self, model, thresholds: Thresholds = None, debug: bool = False
    ) -> None:
        self._model = model
        self._thresholds = thresholds
        # mse that keeps the batch size
        self._mse = lambda a, b: tf.math.reduce_mean(
            tf.math.squared_difference(a, b), axis=[1, 2, 3]
        )
        self._debug = debug
        self._is_classifier = (
            len(self._model.call(inputs=tf.zeros((1, 64, 64, 1)), training=False).shape)
            == 2
        )

    @staticmethod
    def preprocess(crop: np.array) -> tf.Tensor:
        """The preprocessing operation to apply, before feeding the image to the classifier.
        Args:
            crop (np.array): a BGR image. np.uin8
        Return:
            face (tf.Tensor): the post-processed face, tf.float32, 64x64x1
        """

        crop = tf.convert_to_tensor(crop)
        rgb = tf.reverse(crop, axis=[-1])  # 0,255
        gray = tf.image.rgb_to_grayscale(rgb)

        ## NOTE: mandatory convert the image to float32 before
        # calling resize. Resize is stupid and changes the dtype
        # WITHOUT scaling the values in the dtype range-
        # Convert to [0,1]
        gray = tf.image.convert_image_dtype(gray, tf.float32)
        gray = tf.image.resize(gray, (64, 64))

        if tf.equal(tf.rank(gray), 3):
            gray = tf.expand_dims(gray, axis=[0])
        return gray

    @property
    def model(self):
        """Get the model currently in use."""
        return self._model

    @property
    def thresholds(self) -> Thresholds:
        """Get the thresholds currently in use."""
        return self._thresholds

    @staticmethod
    def normalize(image: tf.Tensor) -> tf.Tensor:
        """Given image in [-1,1] returns image in [0,1]."""
        return (image + 1.0) / 2.0

    def __call__(self, face: tf.Tensor) -> ClassificationResult:
        """Using the model and the thresholds (if any), do the classifcation of the face.
        Args:
            face (tf.Tensor): the cropped tensor (use Classifier.preprocess)
        Return:
            ClassificationResult: the result of the classification.
        """

        # if needed, ad batch size
        if tf.equal(tf.rank(face), 3):
            face = tf.expand_dims(face, axis=[0])

        classified = np.array(
            [ClassificationResult.HEADPHONES_OFF] * tf.shape(face)[0].numpy()
        )

        if self._is_classifier:
            predictions = self._model.call(face, training=False)
            classified[
                np.argmax(predictions, axis=-1) == 1
            ] = ClassificationResult.HEADPHONES_ON
        else:
            reconstruction = self._model.call(face, training=False)
            mse = self._mse(face, reconstruction).numpy()

            on_sigma = self._thresholds.on["variance"]

            classified[
                mse <= (self._thresholds.on["mean"] + 3 * on_sigma)
            ] = ClassificationResult.HEADPHONES_ON

        if self._debug:
            for idx, element in enumerate(classified):
                if element != ClassificationResult.UNKNOWN:
                    if self._is_classifier:
                        logging.info("Classified as: %s", element)
                    else:
                        logging.info("Classified as: %s with mse %f", element, mse[idx])
                        # tf.reverse to go from RGB to BGR
                        cv2.imshow(
                            "reconstruction",
                            tf.squeeze(
                                tf.image.convert_image_dtype(
                                    reconstruction[idx], tf.uint8
                                ),
                                axis=[-1],
                            ).numpy(),
                        )

                    cv2.imshow(
                        "input",
                        tf.squeeze(
                            tf.image.convert_image_dtype(face[idx], tf.uint8), axis=[-1]
                        ).numpy(),
                    )
        return classified

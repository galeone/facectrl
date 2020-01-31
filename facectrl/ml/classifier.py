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


class ClassificationResult(Enum):
    """A possible classification result."""

    HEADPHONES_ON = auto()
    HEADPHONES_OFF = auto()
    UNKNOWN = auto()


class Classifier:
    """The Classifier object implements all the logic to trigger a detection.
    Args:
        autoencoder (tf.keras.Model): the previously trained autoencoder.
        thresholds (dict): the dictionary containing the learned thresholds
                           (model selection result).
        debug (bool): when True, it enables the opencv visualization.
    """

    def __init__(
        self, autoencoder: tf.keras.Model, thresholds: Dict, debug: bool = False
    ) -> None:
        self._autoencoder = autoencoder
        self._thresholds = thresholds
        self._mse = tf.keras.losses.MeanSquaredError()
        self._debug = debug

    @staticmethod
    def preprocess(crop: np.array) -> tf.Tensor:
        """The preprocessing operation to apply, before feeding the image to the classifier.
        Args:
            crop (np.array): a BGR image.
        Return:
            face (tf.Tensor): the post-processed face, float32, 64x64. In RGB.
        """
        crop = tf.convert_to_tensor(crop)
        rgb = tf.reverse(crop, axis=[-1])  # 0,255
        # Convert to [0,1]
        rgb = tf.image.convert_image_dtype(rgb, tf.float32)
        # Convert to [-1,1]
        rgb = (rgb - 0.5) * 2.0
        return tf.expand_dims(tf.image.resize(rgb, (64, 64)), axis=[0],)

    @property
    def autoencoder(self) -> tf.keras.Model:
        """Get the autoencoder currently in use."""
        return self._autoencoder

    @property
    def thresholds(self) -> Dict:
        """Get the thresholds currently in use."""
        return self._thresholds

    @staticmethod
    def normalize(image: tf.Tensor) -> tf.Tensor:
        """Given image in [-1,1] returns image in [0,1]."""
        return (image + 1.0) / 2.0

    def __call__(self, face: tf.Tensor) -> ClassificationResult:
        """Using the autoencoder and the thresholds, do the classifcation of the face.
        Args:
            face (tf.Tensor): the cropped tensor (use Classifier.preprocess)
        Return:
            ClassificationResult: the result of the classification.
        """
        classified = ClassificationResult.UNKNOWN
        reconstruction = self._autoencoder(
            face
        )  # face and reconstructions have values in [-1,1]
        mse = self._mse(face, reconstruction).numpy()

        on_sigma = self._thresholds["on_variance"]
        off_sigma = self._thresholds["off_variance"]

        if mse - on_sigma - self._thresholds["LD"] <= self._thresholds["on"]:
            classified = ClassificationResult.HEADPHONES_ON

        if mse + off_sigma >= self._thresholds["off"]:
            classified = ClassificationResult.HEADPHONES_OFF

        if classified != ClassificationResult.UNKNOWN:
            logging.info("Classified as: %s with mse %f", classified, mse)
            if self._debug:
                # tf.reverse to go from RGB to BGR
                cv2.imshow(
                    "reconstruction",
                    tf.squeeze(
                        tf.image.convert_image_dtype(
                            tf.reverse(self.normalize(reconstruction), axis=[-1]),
                            tf.uint8,
                        )
                    ).numpy(),
                )
                cv2.imshow(
                    "input",
                    tf.squeeze(
                        tf.image.convert_image_dtype(
                            tf.reverse(self.normalize(face), axis=[-1]), tf.uint8
                        )
                    ).numpy(),
                )
        else:
            logging.info(
                "Unable to classify the input. mse %s is outside of positive %f and negative %f",
                mse,
                self._thresholds["on"],
                self._thresholds["off"],
            )

        return classified

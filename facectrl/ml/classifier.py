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
    ):
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
        blue, green, red = tf.unstack(crop, axis=-1)
        rgb = tf.stack([red, green, blue])
        face = tf.expand_dims(
            tf.image.resize(tf.image.convert_image_dtype(rgb, tf.float32), (64, 64)),
            axis=[0],
        )
        return face

    @property
    def autoencoder(self):
        """Get the autoencoder currently in use."""
        return self._autoencoder

    @property
    def thresholds(self):
        """Get the thresholds currently in use."""
        return self._thresholds

    def __call__(self, face: tf.Tensor) -> ClassificationResult:
        """Using the autoencoder and the thresholds, do the classifcation of the face.
        Args:
            face (tf.Tensor): the cropped tensor (use Classifier.preprocess)
        Return:
            ClassificationResult: the result of the classification.
        """
        classified = ClassificationResult.UNKNOWN
        reconstruction = self._autoencoder(face)
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
                cv2.imshow(
                    "reconstruction",
                    tf.squeeze(
                        tf.image.convert_image_dtype(reconstruction, tf.uint8)
                    ).numpy(),
                )
                cv2.imshow(
                    "input",
                    tf.squeeze(tf.image.convert_image_dtype(face, tf.uint8)).numpy(),
                )
        else:
            logging.info(
                "Unable to classify the input. mse %s is outside of positive %f and negative %f",
                mse,
                self._thresholds["on"],
                self._thresholds["off"],
            )

        return classified

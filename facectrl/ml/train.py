# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
The package contain the autoencoder (model) definition, the dataset creation,
the metrics, and the training loop. Using AshPy we have model selection for free.
"""


import operator
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict

import ashpy
import tensorflow as tf
from ashpy.losses.classifier import ClassifierLoss
from ashpy.modes import LogEvalMode
from ashpy.restorers.classifier import ClassifierRestorer
from ashpy.trainers.classifier import ClassifierTrainer

BATCH_SIZE = 32
EPOCHS = 50


class ReconstructionLoss(ashpy.metrics.Metric):
    """Computes the Reconstruction Loss (MSE) using the passed dataset."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        model_selection_operator: Callable = None,
        logdir: str = os.path.join(os.getcwd(), "log"),
        name: str = "ReconstructionLoss",
    ) -> None:
        """
        Initialize the Metric.

        Args:
            dataset: the dataset of negatives.
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an `model_selection_operator` is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.
        """
        super().__init__(
            name=name,
            metric=tf.keras.metrics.Mean(),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._mse = tf.keras.losses.MeanSquaredError()
        self._dataset = dataset
        self.mean, self.variance = None, None

    def update_state(self, context: ashpy.contexts.ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """
        values = []
        for images, _ in self._dataset:
            reconstructions = context.classifier_model(
                images, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            mse = self._mse(images, reconstructions)
            values.append(mse)
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(mse)
            )

        self.mean, self.variance = tf.nn.moments(tf.stack(values), axes=[0])


class LD(ashpy.metrics.Metric):
    """Computes the LD (loss discrepancy) using the passed dataset."""

    def __init__(
        self,
        positive_dataset: tf.data.Dataset,
        negative_dataset: tf.data.Dataset,
        model_selection_operator: Callable = None,
        logdir: str = os.path.join(os.getcwd(), "log"),
    ) -> None:
        """
        Initialize the Metric.

        Args:
            dataset: the dataset, of positives, to use.
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an `model_selection_operator` is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.
        """
        super().__init__(
            name="LD",
            metric=tf.keras.metrics.Mean(),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._mse = tf.keras.losses.MeanSquaredError()
        self._mean_positive_loss = ReconstructionLoss(
            positive_dataset, logdir=logdir, name="PositiveLoss"
        )
        self._mean_negative_loss = ReconstructionLoss(
            negative_dataset, logdir=logdir, name="NegativeLoss"
        )
        self._positive_dataset = positive_dataset
        self._negative_dataset = negative_dataset

        self._positive_th = -1.0
        self._negative_th = -1.0
        self._positive_variance = 0.0
        self._negative_variance = 0.0

    def update_state(self, context: ashpy.contexts.ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """
        self._mean_negative_loss.update_state(context)
        self._mean_positive_loss.update_state(context)

        self._distribute_strategy.experimental_run(
            lambda: self._metric.update_state(
                tf.math.abs(
                    self._mean_negative_loss.result()
                    - self._mean_positive_loss.result()
                )
            )
        )
        self._positive_th = self._mean_positive_loss.result()
        self._positive_variance = self._mean_positive_loss.variance.numpy()

        self._negative_th = self._mean_negative_loss.result()
        self._negative_variance = self._mean_negative_loss.variance.numpy()

        self._mean_negative_loss.reset_states()
        self._mean_positive_loss.reset_states()

    def json_write(self, filename: str, what_to_write: Dict) -> None:
        # the json_write function is called when the model selection operation is triggered
        # in this case, it means that there is a great different between the loss value in positive
        # and negativa values. Thus,we want to save the difference (passed in what_to_write)
        # and also the thresholds that we saved in the self._positive_th and self._negative_th

        super().json_write(
            filename,
            {
                **what_to_write,
                **{
                    "positive_threshold": self._positive_th,
                    "positive_variance": self._positive_variance,
                    "negative_threshold": self._negative_th,
                    "negative_variance": self._negative_variance,
                },
            },
        )


def get_model() -> tf.keras.Model:
    """Create a new autoencoder tf.keras.Model."""
    # encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.Dense(128, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dense(64, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dense(32, activation=tf.nn.relu)(net)  # encoding
    net = tf.keras.layers.Dense(64, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dense(128, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dense(64 * 64 * 3, activation=tf.nn.sigmoid)(net)
    reconstructions = tf.keras.layers.Reshape((64, 64, 3))(net)

    model = tf.keras.Model(inputs=inputs, outputs=reconstructions)
    return model


def _to_image(filename: str) -> tf.Tensor:
    """Read the image from the path, and returns the resizes (64x64) image."""

    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (64, 64))
    return image


def _to_ashpy_format(image: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Given an image, returns the pair (image, image)."""
    # ashpy expects the format: features, labels.
    # Since we are traingn an autoencoder, and we want to
    # minimize the reconstruction error, we can pass image as labels
    # and use the classifierloss wit the mse loss inside.
    return image, image


def _augment(image: tf.Tensor) -> tf.Tensor:
    """Given an image, returns a batch with all the tf.image.random*
    transformation applied.
    Args:
        image (tf.Tensor): the input image, 3D tensor.

    Returns:
        images (tf.Tensor): 4D tensor.
    """
    return tf.stack(
        [
            tf.image.random_flip_left_right(image),
            tf.image.random_contrast(image, lower=0.1, upper=0.6),
            tf.image.random_hue(image, max_delta=0.2),
            tf.image.random_brightness(image, max_delta=0.2),
            tf.image.random_saturation(image, lower=0.1, upper=0.6),
            tf.image.random_jpeg_quality(
                image, min_jpeg_quality=20, max_jpeg_quality=100
            ),
        ]
    )


def _build_dataset(glob_path: Path, augmentation: bool = False) -> tf.data.Dataset:
    """Read all the images in the glob_path, and optionally applies
    the data agumentation step.
    Args:
        glob_path (Path): the path of the captured images, with a globa pattern.
        augmentation(bool): when True, applies data agumentation and increase the
                            dataset size.
    Returns:
        the tf.data.Dataset
    """
    dataset = tf.data.Dataset.list_files(str(glob_path)).map(_to_image)
    if augmentation:
        dataset = dataset.map(_augment).unbatch().shuffle(100)

    return dataset.map(_to_ashpy_format).cache().batch(BATCH_SIZE).prefetch(1)


def train(dataset_path: Path, logdir: Path):
    """Train the model obtained with get_model().
    Args:
        dataset_path: path of the dataset containing the headphone on/off pics.
        logdir: destination of the logging dir, checkpoing and selected best models.
    """

    keys = ["on", "off"]
    training_datasets = {
        key: _build_dataset(dataset_path / key / "*.png", augmentation=True)
        for key in keys
    }
    validation_datasets = {
        key: _build_dataset(dataset_path / key / "*.png", augmentation=False)
        for key in keys
    }

    reconstruction_error = ClassifierLoss(tf.keras.losses.MeanSquaredError())
    autoencoder = get_model()
    autoencoder.summary()

    trainer = ClassifierTrainer(
        model=autoencoder,
        # we are intrested only in the performance on unseen data.
        # Thus we measure the metrics on the validation datasets only
        # The only metric that is reallu measure both in training and validation
        # is the loss (executed automatically by ashpy)
        metrics=[
            LD(
                validation_datasets["on"],
                validation_datasets["off"],
                model_selection_operator=operator.gt,
            ),
            ReconstructionLoss(
                validation_datasets["on"], logdir=logdir, name="PositiveLoss"
            ),
            ReconstructionLoss(
                validation_datasets["off"], logdir=logdir, name="NegativeLoss"
            ),
        ],
        optimizer=tf.optimizers.Adam(1e-4),
        loss=reconstruction_error,
        logdir=str(logdir / "on"),
        epochs=EPOCHS,
    )

    trainer(training_datasets["on"], validation_datasets["on"])

    # Restore the best model and save it as a SavedModel
    best_path = logdir / "on" / "best" / "LD"
    autoencoder = ClassifierRestorer(str(best_path)).restore_model(autoencoder)

    dest_path = logdir / "on" / "saved"
    autoencoder.save(str(dest_path))
    copyfile(best_path / "LD.json", dest_path / "LD.json")


def main():
    """Main method, invoked with python -m facectrl.ml.train"""
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist!")

    if (
        not Path(dataset_path / "on").exists()
        or not Path(dataset_path / "off").exists()
    ):
        raise ValueError(f"Wrong dataset {dataset_path}. Missing on/off folders")

    train(dataset_path, Path(args.logdir))


if __name__ == "__main__":
    sys.exit(main())

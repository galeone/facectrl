# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Machine learning models used to detect if you're wearning headphones.
The package contains everything needed: models, training loop and dataset creation.
"""

import operator
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict

import tensorflow as tf

import ashpy
from ashpy.losses.classifier import ClassifierLoss
from ashpy.modes import LogEvalMode
from ashpy.trainers.classifier import ClassifierTrainer

BATCH_SIZE = 50
EPOCHS = 1000


class ReconstructionLoss(ashpy.metrics.Metric):
    """Computes the LD using the passed dataset."""

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

    def update_state(self, context: ashpy.contexts.ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """
        for images, _ in self._dataset:
            reconstructions = context.classifier_model(
                images, training=context.log_eval_mode == LogEvalMode.TRAIN
            )
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(self._mse(images, reconstructions))
            )


class LD(ashpy.metrics.Metric):
    """Computes the LD using the passed dataset."""

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
            dataset: the dataset (of opsitives and negatives) to use.
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
        self._negative_th = self._mean_negative_loss.result()

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
                    "negative_threshold": self._negative_th,
                },
            },
        )


def get_model():
    """Create a new autoencoder tf.keras.Model."""
    # encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)(net)
    net = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(net)
    net = tf.keras.layers.Dense(32, activation=tf.nn.sigmoid)(net)  # encoding
    net = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(net)
    net = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)(net)
    net = tf.keras.layers.Dense(64 * 64 * 3, activation=tf.nn.sigmoid)(net)
    reconstructions = tf.keras.layers.Reshape((64, 64, 3))(net)

    model = tf.keras.Model(inputs=inputs, outputs=reconstructions)
    return model


def _define_or_restore(
    logdir: Path,
    positive_dataset: tf.data.Dataset = None,
    negative_dataset: tf.data.Dataset = None,
):
    reconstruction_error = ClassifierLoss(tf.keras.losses.MeanSquaredError())
    autoencoder = get_model()
    autoencoder.summary()

    # terrible hack, refacator
    metrics = (
        []
        if not positive_dataset
        else [
            LD(
                positive_dataset, negative_dataset, model_selection_operator=operator.gt
            ),
            ReconstructionLoss(positive_dataset, logdir=logdir, name="PositiveLoss"),
            ReconstructionLoss(negative_dataset, logdir=logdir, name="NegativeLoss"),
        ]
    )
    trainer = ClassifierTrainer(
        model=autoencoder,
        metrics=metrics,
        optimizer=tf.optimizers.Adam(1e-4),
        loss=reconstruction_error,
        logdir=str(logdir),
        epochs=EPOCHS,
    )
    return trainer, autoencoder


def _to_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (64, 64))
    return image


def _to_ashpy_format(image):
    # ashpy expects the format: features, labels.
    # Sicne we are traingn an autoencoder, and we want to
    # minimize the reconstruction error, we can pass image as labels
    # and use the classifierloss wit the mse loss inside.
    return image, image


def _train(
    positive_dataset: tf.data.Dataset, negative_dataset: tf.data.Dataset, logdir: Path
):

    trainer, _ = _define_or_restore(logdir, positive_dataset, negative_dataset)
    trainer(positive_dataset, positive_dataset)


def train(dataset_path: Path, logdir: Path):
    """Train the model obtained with get_model().
    Args:
        dataset_path: path of the dataset containing the headphone on/off pics.
        logdir: destination of the logging dir, checkpoing and selected best models.
    """

    datasets = {
        key: tf.data.Dataset.list_files(str(dataset_path / key / "*.png"))
        .map(_to_image)
        .map(_to_ashpy_format)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(1)
        for key in ["on", "off"]
    }

    _train(
        positive_dataset=datasets["on"],
        negative_dataset=datasets["off"],
        logdir=logdir / "on",
    )

    # Keep the best model checkpoints and export them as SavedModels
    key = "on"
    # Use the trainer to correctly restore the parameters of the best model
    best_path = logdir / key / "best" / "LD"
    _, autoencoder = _define_or_restore(best_path)

    dest_path = logdir / key / "saved"
    autoencoder.save(str(dest_path))
    copyfile(best_path / "LD.json", dest_path / "LD.json")


def main():
    """Main method, invoked with python -m facectrl.ml"""
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

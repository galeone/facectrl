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
from typing import Callable

import tensorflow as tf

import ashpy
from ashpy.losses.classifier import ClassifierLoss
from ashpy.models.convolutional.autoencoders import Autoencoder
from ashpy.modes import LogEvalMode
from ashpy.trainers.classifier import ClassifierTrainer

BATCH_SIZE = 50


class AUC(ashpy.metrics.Metric):
    """Computes the AUC using the passed dataset."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
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
            name="AUC",
            metric=tf.keras.metrics.AUC(num_thresholds=100),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        # this is not the loss (single scalar value)
        # this is the MSE for each sample in the batch
        self._mse = lambda y_true, y_pred: tf.reduce_mean(
            tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3]
        )
        self._dataset = dataset

    def update_state(self, context: ashpy.contexts.ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """

        for images, labels in self._dataset:
            reconstructions = context.classifier_model(
                images, training=context.log_eval_mode == LogEvalMode.TRAIN
            )
            scores = self._mse(images, reconstructions)
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(labels, scores)
            )


def get_model():
    """Create a new autoencoder tf.keras.Model."""
    autoencoder = Autoencoder(
        (64, 64),
        (4, 4),
        kernel_size=3,
        initial_filters=32,
        filters_cap=64,
        encoding_dimension=32,
        channels=3,
    )

    # encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    _, reconstruction = autoencoder(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=reconstruction)
    return model


def _define_or_restore(logdir: Path, full_dataset: tf.data.Dataset = None):
    reconstruction_error = ClassifierLoss(tf.keras.losses.MeanSquaredError())
    autoencoder = get_model()

    # terrible hack, refacator
    metrics = (
        []
        if not full_dataset
        else [AUC(full_dataset, model_selection_operator=operator.gt)]
    )
    trainer = ClassifierTrainer(
        model=autoencoder,
        metrics=metrics,
        optimizer=tf.optimizers.Adam(1e-4),
        loss=reconstruction_error,
        logdir=str(logdir),
        epochs=250,
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
    positive_dataset: tf.data.Dataset,
    negative_dataset: tf.data.Dataset,
    training_positive: bool,
    logdir: Path,
):
    # change the dataset format: from image, image, that is OK for trianing the autoencoder
    # to image, label that's needed to measure the AUC.
    # Moreover, if we are training the model of positives, the 1 label must be on positives
    # If instead, we are traiing the model of negativesl the 1 label must be on negatives

    # The labels are needed only for measuring the AUC, therefore we create a separate dataset

    positive_label = 1
    negative_label = 0
    training_dataset = positive_dataset
    if not training_positive:
        positive_label = 0
        negative_label = 1
        training_dataset = negative_dataset

    positive_dataset = positive_dataset.unbatch().map(
        lambda image, label_image: (image, positive_label)
    )
    negative_dataset = negative_dataset.unbatch().map(
        lambda image, label_image: (image, negative_label)
    )

    full_dataset = (
        positive_dataset.concatenate(negative_dataset)
        .shuffle(buffer_size=100)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )
    trainer, _ = _define_or_restore(logdir, full_dataset)
    trainer(training_dataset, training_dataset)


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
        training_positive=True,
    )
    _train(
        positive_dataset=datasets["on"],
        negative_dataset=datasets["off"],
        logdir=logdir / "off",
        training_positive=False,
    )

    # Keep the best model checkpoints and export them as SavedModels
    for key in ["on", "off"]:
        # Use the trainer to correctly restore the parameters of the best model
        best_path = logdir / key / "best" / "AUC"
        _, autoencoder = _define_or_restore(best_path)

        dest_path = logdir / key / "saved"
        autoencoder.save(str(dest_path))
        copyfile(best_path / "AUC.json", dest_path / "AUC.json")


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

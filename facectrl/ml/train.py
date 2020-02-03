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
        self.inputs, self.reconstructions = None, None

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
        # save the last batch
        self.inputs = images
        self.reconstructions = reconstructions

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
        """The json_write function is called when the model selection operation
        is performed. In this case, it means that there is a great different
        between the loss value in positive and negativa values.
        Thus,we want to save the difference (passed in what_to_write) and also the
        thresholds that we saved in the self._positive_th and self._negative_th.

        Args:
            filename: the path of the JSON
            what_to_write: the dictionary with the values to write on the JSON.
        """

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

    def log(self, step: int) -> None:
        """
        Log the metric.
        Args:
            step: global step of training
        """
        super().log(step)
        tf.summary.image(
            "positive",
            _normalize(
                tf.concat(
                    [
                        self._mean_positive_loss.inputs,
                        self._mean_positive_loss.reconstructions,
                    ],
                    axis=2,
                )
            ),
            step=step,
        )

        tf.summary.image(
            "negative",
            _normalize(
                tf.concat(
                    [
                        self._mean_negative_loss.inputs,
                        self._mean_negative_loss.reconstructions,
                    ],
                    axis=2,
                )
            ),
            step=step,
        )


def get_model() -> tf.keras.Model:
    """Create a new autoencoder tf.keras.Model."""
    # encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.Dense(64)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)

    net = tf.keras.layers.Dense(32)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)

    net = tf.keras.layers.Dense(16)(net)  # encoding
    net = tf.keras.layers.Activation(tf.nn.relu)(net)

    net = tf.keras.layers.Dense(32)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)

    net = tf.keras.layers.Dense(64)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)

    net = tf.keras.layers.Dense(64 * 64 * 3)(net)
    net = tf.keras.layers.Activation(tf.nn.tanh)(net)
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


def _squash(image: tf.Tensor) -> tf.Tensor:
    """Given an image in [0,1] squash its values in [-1,1]"""
    return (image - 0.5) * 2.0  # [-1, 1] range


def _normalize(image: tf.Tensor) -> tf.Tensor:
    """Given an image in [-1,1] squash its values in [0,1]"""
    return (image + 1.0) / 2.0


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


def _build_dataset(
    glob_path: Path, batch_size, augmentation: bool = False
) -> tf.data.Dataset:
    """Read all the images in the glob_path, and optionally applies
    the data agumentation step.
    Args:
        glob_path (Path): the path of the captured images, with a glob pattern.
        batch_size (int): the batch size.
        augmentation(bool): when True, applies data agumentation and increase the
                            dataset size.
    Returns:
        the tf.data.Dataset
    """
    dataset = tf.data.Dataset.list_files(str(glob_path)).map(_to_image)
    if augmentation:
        dataset = dataset.map(_augment).unbatch()
    dataset = dataset.map(_squash)  # in [-1, 1] range
    dataset = dataset.map(_to_ashpy_format).cache()
    # agumentation == training -> shuffle the elemenents of the new dataset
    # after the .cache() step
    if augmentation:
        dataset = dataset.shuffle(100)
    return dataset.batch(batch_size).prefetch(1)


def train(dataset_path: Path, batch_size: int, epochs: int, logdir: Path) -> None:
    """Train the model obtained with get_model().
    Args:
        dataset_path (Path): path of the dataset containing the headphone on/off pics.
        batch_size (int): the batch size.
        epochs (int): number of the training epochs.
        logdir (Path): destination of the logging dir, checkpoing and selected best models.
    """

    keys = ["on", "off"]
    training_datasets = {
        key: _build_dataset(dataset_path / key / "*.png", batch_size, augmentation=True)
        for key in keys
    }
    validation_datasets = {
        key: _build_dataset(
            dataset_path / key / "*.png", batch_size, augmentation=False
        )
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
        epochs=epochs,
    )

    trainer(training_datasets["on"], validation_datasets["on"])

    # Restore the best model and save it as a SavedModel
    best_path = logdir / "on" / "best" / "LD"
    autoencoder = ClassifierRestorer(str(best_path)).restore_model(autoencoder)

    dest_path = logdir / "on" / "saved"
    autoencoder.save(str(dest_path))
    copyfile(best_path / "LD.json", dest_path / "LD.json")


def main() -> int:
    """Main method, invoked with python -m facectrl.ml.train"""
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist!")

    if (
        not Path(dataset_path / "on").exists()
        or not Path(dataset_path / "off").exists()
    ):
        raise ValueError(f"Wrong dataset {dataset_path}. Missing on/off folders")

    train(dataset_path, args.batch_size, args.epochs, Path(args.logdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

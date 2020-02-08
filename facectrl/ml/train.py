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
import numpy as np
import tensorflow as tf
from ashpy.contexts import ClassifierContext
from ashpy.losses.classifier import ClassifierLoss
from ashpy.losses.executor import Executor
from ashpy.modes import LogEvalMode
from ashpy.restorers.classifier import ClassifierRestorer
from ashpy.trainers.classifier import ClassifierTrainer

from facectrl.ml.classifier import ClassificationResult, Classifier, Thresholds
from facectrl.ml.model import CVAE


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
        self.mean, self.variance = tf.nn.moments(tf.zeros((1, 1)), axes=[0])
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
            reconstructions = context.classifier_model.reconstruct(images)

            mse = self._mse(images, reconstructions)
            values.append(mse)
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(mse)
            )
        # save the last batch
        self.inputs = images
        self.reconstructions = reconstructions

        self.mean, self.variance = tf.nn.moments(tf.stack(values), axes=[0])


class AEAccuracy(ashpy.metrics.Metric):
    """Computes the AutoEncoder (AE) classification accuracy using the passed datasets."""

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
            name="AEAccuracy",
            metric=tf.keras.metrics.Accuracy(),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._mse = tf.keras.losses.MeanSquaredError()
        self._mean_positive_loss = ReconstructionLoss(
            positive_dataset, logdir=logdir, name="mse/positive"
        )
        self._mean_negative_loss = ReconstructionLoss(
            negative_dataset, logdir=logdir, name="mse/negative"
        )
        self._positive_dataset = positive_dataset
        self._negative_dataset = negative_dataset

        self._thresholds = Thresholds(
            on={
                "mean": self._mean_positive_loss.result(),
                "variance": self._mean_negative_loss.variance.numpy(),
            },
            off={
                "mean": self._mean_negative_loss.result(),
                "variance": self._mean_negative_loss.variance.numpy(),
            },
        )
        batch_size = next(iter(self._positive_dataset.take(1)))[0].shape[0]

        def _positive(x, y):
            return x, 1

        def _negative(x, y):
            return x, 0

        self._full_dataset = (
            self._positive_dataset.unbatch()
            .map(_positive)
            .concatenate(self._negative_dataset.unbatch().map(_negative))
            .shuffle(100)
            .batch(batch_size)
        )

    def update_state(self, context: ashpy.contexts.ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """

        # Compute the loss on positive and negative dataset
        self._mean_negative_loss.update_state(context)
        self._mean_positive_loss.update_state(context)

        # Use the mean and variance to classify the full dataset
        self._thresholds = Thresholds(
            on={
                "mean": self._mean_positive_loss.result(),
                "variance": self._mean_negative_loss.variance.numpy(),
            },
            off={
                "mean": self._mean_negative_loss.result(),
                "variance": self._mean_negative_loss.variance.numpy(),
            },
        )

        classifier = Classifier(
            cvae=context.classifier_model, thresholds=self._thresholds
        )
        for images, y_true in self._full_dataset:
            reconstructions = context.classifier_model.reconstruct(images)
            y_pred = classifier(reconstructions)
            y_pred[y_pred == ClassificationResult.HEADPHONES_ON] = 1
            y_pred[y_pred == ClassificationResult.HEADPHONES_OFF] = 0
            y_pred[y_pred == ClassificationResult.UNKNOWN] = -1
            y_pred = tf.stack(y_pred)
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(y_true, y_pred)
            )

        # reset the states of the ReconstructionLoss metrics
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
            filename, {**what_to_write, **self._thresholds.asdict()},
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

        tf.summary.scalar("positive/mean", self._thresholds.on["mean"], step=step)
        tf.summary.scalar(
            "positive/variance", self._thresholds.on["variance"], step=step
        )
        tf.summary.scalar("negative/mean", self._thresholds.off["mean"], step=step)
        tf.summary.scalar(
            "negative/variance", self._thresholds.off["variance"], step=step
        )


class MaximizeELBO(Executor):
    r"""Maximizes the Evidence Lowe BOund (ELBO)"""

    def __init__(self) -> None:
        r"""
        Initialize :py:class:`MaximizeELBO`.

        Returns:
            :py:obj:`None`
        """
        super().__init__()

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    @Executor.reduce_loss
    def call(
        self, context: ClassifierContext, *, features: tf.Tensor, **kwargs,
    ) -> tf.Tensor:
        r"""
        Compute the loss.
        Args:
            context (:py:class:`ashpy.ClassifierContext`): Context for classification.
            features (:py:class:`tf.Tensor`): Inputs for the VAE.
            **kwargs:
        Returns:
            :py:class:`tf.Tensor`: Loss value.
        """
        mean, logvar = context.classifier_model.encode(features)
        z = context.classifier_model.reparameterize(mean, logvar)
        reconstructions = context.classifier_model.decode(z)

        mse = tf.reduce_mean(
            tf.math.squared_difference(reconstructions, features), axis=[1, 2, 3]
        )
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        loss_per_element = -1.0 * (logpz - logqz_x) + mse  # (batch_size,)
        # Support for AshPy distribute computation (reduction strategy)
        # loss has shape [batch_size], we need [batch_size, 1]
        loss = tf.expand_dims(loss_per_element, axis=-1)
        return tf.reduce_mean(loss, axis=[1], keepdims=True)


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
    # Since we are traingn a cvae, and we want to
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
    """Train the CVAE model.
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

    cvae = CVAE()
    # define the model by passing a dummy input
    # the call method of the CVAE is the reconstruction
    # autoencoder-lik
    cvae(tf.zeros((1, 64, 64, 3)))
    cvae.summary()

    trainer = ClassifierTrainer(
        model=cvae,
        # we are intrested only in the performance on unseen data.
        # Thus we measure the metrics on the validation datasets only
        # The only metric that is reallu measure both in training and validation
        # is the loss (executed automatically by ashpy)
        metrics=[
            AEAccuracy(
                validation_datasets["on"],
                validation_datasets["off"],
                model_selection_operator=operator.gt,
            )
        ],
        optimizer=tf.optimizers.Adam(1e-4),
        loss=MaximizeELBO(),
        logdir=str(logdir / "on"),
        epochs=epochs,
    )

    trainer(training_datasets["on"], validation_datasets["on"])

    # Restore the best model and save it as a SavedModel
    best_path = logdir / "on" / "best" / "AEAccuracy"
    cvae = ClassifierRestorer(str(best_path)).restore_model(cvae)

    dest_path = logdir / "on" / "saved"
    cvae.save(str(dest_path))
    copyfile(best_path / "AEAccuracy.json", dest_path / "AEAccuracy.json")


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

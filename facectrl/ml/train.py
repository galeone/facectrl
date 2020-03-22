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
from glob import glob
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict

import tensorflow as tf

import ashpy
from ashpy.contexts import ClassifierContext
from ashpy.losses.classifier import ClassifierLoss
from ashpy.losses.executor import Executor
from ashpy.metrics.classifier import ClassifierMetric
from ashpy.restorers.classifier import ClassifierRestorer
from ashpy.trainers.classifier import ClassifierTrainer
from facectrl.ml.classifier import ClassificationResult, Classifier, Thresholds
from facectrl.ml.model import AE, NN, VAE


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

        self._mse = lambda x, y: tf.reduce_mean(
            tf.math.squared_difference(x, y), axis=[1, 2, 3]
        )
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
            reconstructions = context.classifier_model.reconstruct(
                images, training=False
            )

            mse = self._mse(images, reconstructions)
            values.extend(mse)
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

        self._full_dataset = (
            self._positive_dataset.unbatch()
            .map(self._positive)
            .concatenate(self._negative_dataset.unbatch().map(self._negative))
            .batch(batch_size)
        )

    @staticmethod
    def _positive(x, y):
        return x, 1

    @staticmethod
    def _negative(x, y):
        return x, 0

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
            model=context.classifier_model, thresholds=self._thresholds
        )
        for images, y_true in self._full_dataset:
            y_pred = classifier(images)
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
            tf.concat(
                [
                    self._mean_positive_loss.inputs,
                    self._mean_positive_loss.reconstructions,
                ],
                axis=2,
            ),
            step=step,
        )

        tf.summary.image(
            "negative",
            tf.concat(
                [
                    self._mean_negative_loss.inputs,
                    self._mean_negative_loss.reconstructions,
                ],
                axis=2,
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

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        return (
            0.5
            * tf.reduce_sum(tf.math.squared_difference(targets, mean))
            / (2 * tf.square(std))
        )

    @staticmethod
    def _kl_gaussian(mean, logvar):
        var = tf.exp(logvar)
        kl = 0.5 * tf.reduce_sum(tf.square(mean) + var - 1.0 - logvar)
        return kl

    # @Executor.reduce_loss
    def call(
        self, context: ClassifierContext, *, features: tf.Tensor, **kwargs,
    ) -> tf.Tensor:
        r"""
        Compute the loss.
        Args:
            context (:py:class:`ashpy.ClassifierContext`): Context for classification.
            features (:py:class:`tf.Tensor`): Inputs for the Model.
            **kwargs:
        Returns:
            :py:class:`tf.Tensor`: Loss value.
        """
        mean, logvar = context.classifier_model.encode(features, training=True)
        z = context.classifier_model.reparameterize(mean, logvar, training=True)
        reconstructions = context.classifier_model.decode(z, training=True)

        kl_loss = self._kl_gaussian(mean, logvar)
        reconstruction_loss = self._gaussian_log_likelihood(
            features, reconstructions, 1e-2
        )

        return kl_loss + 10000 * reconstruction_loss


class DatasetBuilder:
    @staticmethod
    def _to_image(filename: str) -> tf.Tensor:
        """Read the image from the path, and returns the resizes (64x64) image."""

        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (64, 64))
        return image

    @staticmethod
    def _to_autoencoder_format(image: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Given an image, returns the pair (image, image)."""
        return image, image

    @staticmethod
    def augment(image: tf.Tensor) -> tf.Tensor:
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
                tf.image.random_brightness(image, max_delta=0.2),
                tf.image.random_jpeg_quality(
                    image, min_jpeg_quality=20, max_jpeg_quality=100
                ),
            ]
        )

    def __call__(
        self,
        glob_path: Path,
        batch_size: int,
        augmentation: bool = False,
        use_label: int = -1,
        skip: int = -1,
        take: int = -1,
    ) -> tf.data.Dataset:
        """Read all the images in the glob_path, and optionally applies
        the data agumentation step.
        Args:
            glob_path (Path): the path of the captured images, with a glob pattern.
            batch_size (int): the batch size.
            augmentation(bool): when True, applies data agumentation and increase the
                                dataset size.
            skip: how many files to skip from the glob_path listed files. -1 = do not skip.
            take: how many files to take from the glob_path listed files. -1 = take all.
        Returns:
            the tf.data.Dataset
        """
        dataset = tf.data.Dataset.list_files(str(glob_path)).map(self._to_image)

        if skip != -1:
            dataset = dataset.skip(skip)
        if take != -1:
            dataset = dataset.take(take)

        if augmentation:
            dataset = dataset.map(self.augment).unbatch()
            # 4 because augment increase the dataset size by 4
            dataset = dataset.shuffle(4 * len(glob(str(glob_path))))
        if use_label != -1:
            label = use_label
            dataset = dataset.map(lambda image: (image, label))
        else:
            dataset = dataset.map(self._to_autoencoder_format)
        return dataset.batch(batch_size).prefetch(1)


def train(
    dataset_path: Path, batch_size: int, epochs: int, logdir: Path, model_type: str
) -> None:
    """Train the Model.
    Args:
        dataset_path (Path): path of the dataset containing the headphone on/off pics.
        batch_size (int): the batch size.
        epochs (int): number of the training epochs.
        logdir (Path): destination of the logging dir, checkpoing and selected best models.
    """

    is_classifier = model_type == "classifier"
    training_datasets = {}
    validation_datasets = {}

    for key in ["on", "off"]:
        label = -1
        if is_classifier:
            if key == "on":
                label = 1
            elif key == "off":
                label = 0
        training_datasets[key] = DatasetBuilder()(
            glob_path=dataset_path / key / "*.png",
            batch_size=batch_size,
            augmentation=True,
            use_label=label,
            # skip the first 100 files: reserved for validation
            skip=100,
        )

        validation_datasets[key] = DatasetBuilder()(
            glob_path=dataset_path / key / "*.png",
            batch_size=batch_size,
            augmentation=True,
            use_label=label,
            # take the first 100
            take=100,
        )

    if is_classifier:
        model = NN()
        loss = ClassifierLoss(
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        metrics = [
            ClassifierMetric(
                tf.keras.metrics.BinaryAccuracy(), model_selection_operator=operator.gt
            )
        ]
        best_path = logdir / "on" / "best" / "binary_accuracy"
        best_json = "binary_accuracy.json"
        training_dataset = (
            training_datasets["on"]
            .unbatch()
            .concatenate(training_datasets["off"].unbatch())
            # 4 because the training dataset is augmented and the dataset size
            # itself increased by a factor of 4
            .shuffle(4 * len(glob(str(dataset_path / "on" / "*.png"))))
            .batch(batch_size)
            .prefetch(1)
        )

        validation_dataset = (
            validation_datasets["on"]
            .unbatch()
            .concatenate(validation_datasets["off"].unbatch())
            .batch(batch_size)
            .prefetch(1)
        )

    else:
        if model_type == "vae":
            model = VAE()
            loss = MaximizeELBO()
        if model_type == "ae":
            model = AE()
            loss = ClassifierLoss(tf.keras.losses.MeanSquaredError())

        metrics = [
            AEAccuracy(
                validation_datasets["on"],
                validation_datasets["off"],
                model_selection_operator=operator.gt,
            )
        ]
        best_path = logdir / "on" / "best" / "AEAccuracy"
        best_json = "AEAccuracy.json"

        training_dataset = training_datasets["on"]
        validation_dataset = validation_datasets["on"]

    # define the model by passing a dummy input
    # the call method of the Model is the reconstruction
    # autoencoder-lik
    model(tf.zeros((1, 64, 64, 1)), training=True)
    model.summary()

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        5e-3, decay_rate=0.98, decay_steps=1000, staircase=True
    )
    trainer = ClassifierTrainer(
        model=model,
        metrics=metrics,
        optimizer=tf.optimizers.Adam(learning_rate),
        loss=loss,
        logdir=str(logdir / "on"),
        epochs=epochs,
    )

    trainer(training_dataset, validation_dataset, measure_performance_freq=-1)

    # Restore the best model and save it as a SavedModel

    model = ClassifierRestorer(str(best_path)).restore_model(model)
    # Define the input shape by calling it on fake data
    model(tf.zeros((1, 64, 64, 1)), training=False)
    model.summary()

    dest_path = logdir / "on" / "saved"
    tf.saved_model.save(model, str(dest_path))
    copyfile(best_path / best_json, dest_path / best_json)


def main() -> int:
    """Main method, invoked with python -m facectrl.ml.train"""
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--model", type=str, choices=["ae", "vae", "classifier"], required=True
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist!")

    if (
        not Path(dataset_path / "on").exists()
        or not Path(dataset_path / "off").exists()
    ):
        raise ValueError(f"Wrong dataset {dataset_path}. Missing on/off folders")

    train(dataset_path, args.batch_size, args.epochs, Path(args.logdir), args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

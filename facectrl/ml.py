# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Machine learning models used to detect if you're wearning headphones.
The package contains everything needed: models, training loop and dataset creation.
"""

import operator
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import ashpy
import tensorflow as tf
from ashpy.losses.classifier import ClassifierLoss
from ashpy.models.convolutional.autoencoders import Autoencoder
from ashpy.trainers.classifier import ClassifierTrainer


def get_model():
    """Create a new autoencoder tf.keras.Model."""
    autoencoder = Autoencoder(
        (64, 64),
        (4, 4),
        kernel_size=3,
        initial_filters=16,
        filters_cap=64,
        encoding_dimension=50,
        channels=3,
    )

    # encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    _, reconstruction = autoencoder(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=reconstruction)
    return model


def _define_or_restore(logdir: Path):
    reconstruction_error = ClassifierLoss(tf.keras.losses.MeanSquaredError())
    autoencoder = get_model()

    trainer = ClassifierTrainer(
        model=autoencoder,
        optimizer=tf.optimizers.Adam(1e-4),
        loss=reconstruction_error,
        metrics=[ashpy.metrics.ClassifierLoss(model_selection_operator=operator.lt)],
        logdir=str(logdir),
        epochs=250,
    )
    return trainer, autoencoder


def _train(dataset: tf.data.Dataset, logdir: Path):
    trainer, _ = _define_or_restore(logdir)
    trainer(dataset, dataset)


def train(dataset_path: Path, logdir: Path):
    """Train the model obtained with get_model().
    Args:
        dataset_path: path of the dataset containing the headphone on/off pics.
        logdir: destination of the logging dir, checkpoing and selected best models.
    """

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

    datasets = {
        key: tf.data.Dataset.list_files(str(dataset_path / key / "*.png"))
        .map(_to_image)
        .map(_to_ashpy_format)
        .cache()
        .batch(10)
        .prefetch(1)
        for key in ["on", "off"]
    }

    _train(datasets["on"], logdir / "on")
    _train(datasets["off"], logdir / "off")

    # Keep the best model checkpoints and export them as SavedModels
    for key in ["on", "off"]:
        # Use the trainer to correctly restore the parameters of the best model
        best_path = logdir / key / "best" / "loss"
        _, autoencoder = _define_or_restore(best_path)

        dest_path = logdir / key / "saved"
        autoencoder.save(str(dest_path))
        copyfile(best_path / "loss.json", dest_path / "loss.json")


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

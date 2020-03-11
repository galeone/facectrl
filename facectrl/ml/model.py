# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Various models used to tackle this proble.
VAE: variation autoencoder
AE: autoencoder
MLCNN: simple classifier (Multi layer CNN)
"""

import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._latent_dim = 128
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(
                    128, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    64, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    32, (3, 3), padding="same", strides=(2, 2), activation=tf.nn.relu
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self._latent_dim),  # linear
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim)),
                tf.keras.layers.Reshape((4, 4, 8)),
                tf.keras.layers.Conv2DTranspose(
                    32, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    64, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    128, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    3, (3, 3), activation=tf.nn.tanh, padding="same", strides=(2, 2)
                ),  # 64x64x3, [-1,1]
            ]
        )

    def encode(self, x, training):
        return self.encoder(x, training)

    def decode(self, z, training):
        logits = self.decoder(z, training)
        return logits

    def reconstruct(self, x, training):
        return self.decode(self.encode(x, training), training)

    @tf.function
    def call(self, inputs, training):
        return self.reconstruct(inputs, training)


class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._latent_dim = 128
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.Conv2D(
                    64, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.Conv2D(
                    128, (3, 3), padding="same", activation=tf.nn.relu, strides=(2, 2)
                ),  # 128x8x8
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self._latent_dim * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim,)),
                tf.keras.layers.Reshape((4, 4, 8)),
                tf.keras.layers.Conv2DTranspose(
                    128, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.Conv2DTranspose(
                    64, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.Conv2DTranspose(
                    32, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.Conv2DTranspose(
                    3, (3, 3), activation=tf.nn.tanh, padding="same", strides=(2, 2)
                ),  # 64x64x3, [-1,1]
            ]
        )

    def encode(self, x, training):
        mean, logvar = tf.split(self.encoder(x, training), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar, training):
        if training:
            return mean
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, training):
        logits = self.decoder(z, training)
        return logits

    def reconstruct(self, x, training):
        mean, logvar = self.encode(x, training)
        z = self.reparameterize(mean, logvar, training)
        return self.decode(z, training)

    @tf.function
    def call(self, inputs, training):
        return self.reconstruct(inputs, training)


class MLCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._num_classes = 2
        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(
                    128, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    64, (3, 3), activation=tf.nn.relu, padding="same", strides=(2, 2)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    32, (3, 3), padding="same", strides=(2, 2), activation=tf.nn.relu
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self._num_classes),  # linear
            ]
        )

    @tf.function
    def call(self, inputs, training):
        return self._model(inputs, training)

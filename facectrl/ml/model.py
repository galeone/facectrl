# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Various models used to tackle this proble.
VAE: variation autoencoder
AE: autoencoder
NN: simple classifier (Multi layer CNN)
"""

import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._latent_dim = 64
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(self._latent_dim),  # linear
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim)),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64 * 64, activation=tf.nn.sigmoid),
                tf.keras.layers.Reshape((64, 64, 1)),
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
        self._latent_dim = 64
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(self._latent_dim * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim)),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64 * 64, activation=tf.nn.sigmoid),
                tf.keras.layers.Reshape((64, 64, 1)),
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


class NN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(2),
            ]
        )

    @tf.function
    def call(self, inputs, training):
        return self._model(inputs, training)

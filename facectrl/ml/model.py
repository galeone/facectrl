# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
VAE: model definition. 
"""

import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._latent_dim = 32
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(self._latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim,)),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64 * 64 * 3, activation=tf.nn.tanh),
                tf.keras.layers.Reshape((64, 64, 3)),
            ]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    @tf.function
    def call(self, inputs):
        return self.reconstruct(inputs)


class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._latent_dim = 32
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(self._latent_dim * 2),  # mu, sigma
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim,)),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(64 * 64 * 3, activation=tf.nn.tanh),
                tf.keras.layers.Reshape((64, 64, 3)),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def reconstruct(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    @tf.function
    def call(self, inputs):
        return self.reconstruct(inputs)

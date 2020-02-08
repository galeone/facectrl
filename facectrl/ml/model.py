# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Convolutioanl Variational AutoEncoder (VAE) definition.
"""


import tensorflow as tf


class CVAE(tf.keras.Model):
    def __init__(self):
        self._latent_dim = 128
        super().__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.MaxPool2D((2, 2), padding="same"),  # 32x32x16
                tf.keras.layers.Conv2D(
                    8, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.MaxPool2D((2, 2), padding="same"),  # 16x16x8
                tf.keras.layers.Conv2D(4, (3, 3), padding="same"),  # linear
                tf.keras.layers.MaxPool2D((2, 2), padding="same"),  # 8x8x4 -> 256
                tf.keras.layers.Flatten(),  # linear
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._latent_dim,)),
                tf.keras.layers.Reshape((4, 4, 8)),
                tf.keras.layers.Conv2D(
                    4, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.UpSampling2D((2, 2)),  # 8x8x4
                tf.keras.layers.Conv2D(
                    8, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.UpSampling2D((2, 2)),  # 16x16x8
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.UpSampling2D((2, 2)),  # 32x32x16
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation=tf.nn.relu, padding="same"
                ),
                tf.keras.layers.UpSampling2D((2, 2)),  # 64x64x16
                tf.keras.layers.Conv2D(
                    3, (3, 3), activation=tf.nn.tanh, padding="same"
                ),  # 64x64x3, [-1,1]
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

    def call(self, x):
        return self.reconstruct(x)

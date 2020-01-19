# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and executable module to control the player with your face."""

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import gi
import tensorflow as tf

from facectrl.detector import FaceDetector
from facectrl.video import Tracker, WebcamVideoStream


class Controller:
    def __init__(
        self, player, stream: WebcamVideoStream, detector: FaceDetector, logdir: Path,
    ):
        """The controller, that controles the player using the models."""
        self._player = player
        self._stream = stream
        self._detector = detector
        self._models = {
            "on": tf.keras.models.load_model(str(logdir / "on" / "saved")),
            "off": tf.keras.models.load_model(str(logdir / "off" / "saved")),
        }
        self._thresh = {
            "on": float(
                json.load(open(str(logdir / "on" / "saved" / "loss.json")))["loss"]
            ),
            "off": float(
                json.load(open(str(logdir / "off" / "saved" / "loss.json")))["loss"]
            ),
        }
        self._mse = tf.keras.losses.MeanSquaredError()

        self._playing = False

        def _play(player, status):
            self._playing = True

        def _pause(player, status):
            self._playing = True

        self._player.connect("playback-status::playing", _play)
        self._player.connect("playback-status::paused", _pause)

    def start(self):
        with self._stream:
            while True:
                found = False
                # find the face for the first time in this loop
                print("Detecting a face...")
                while not found:
                    frame = self._stream.read()
                    bounding_box = self._detector.detect(frame)
                    if bounding_box[-1] != 0:
                        found = True
                        print("Face found!")

                # extract the face and classify it
                face = tf.expand_dims(
                    tf.image.resize(
                        tf.image.convert_image_dtype(
                            FaceDetector.crop(frame, bounding_box, expansion=(30, 30)),
                            tf.float32,
                        ),
                        (64, 64),
                    ),
                    axis=[0],
                )

                classified = False
                for key in ["on", "off"]:
                    mse = self._mse(face, self._models[key](face))
                    print("mse: ", mse.numpy())
                    if mse <= self._thresh[key]:
                        classified = key
                        break

                if classified:
                    print("Classified as: ", classified, " with mse: ", mse)
                    tracker = Tracker(frame, bounding_box)
                    try:
                        if classified == "on" and not self._playing:
                            player.play()
                        if classified == "off" and self._playing:
                            player.pause()
                        # Start tracking stream
                        while True:
                            frame = self._stream.read()
                            tracker.track(frame)
                    except ValueError:
                        # When the tracking fails, there is a status change
                        if classified == "on":
                            player.pause()

                        if classified == "off":
                            player.play()
                else:
                    print("Unable to classify the input")


def main():
    """Main method, invoked with python -m facectrl.ctrl."""
    gi.require_version("Playerctl", "2.0")
    from gi.repository import Playerctl, GLib  # pylint: disable=import-outside-toplevel

    parser = ArgumentParser()
    parser.add_argument("--player", required=True, type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--classifier-params", required=True)
    parser.add_argument("--stream-source", default=0)
    args = parser.parse_args()

    player = Playerctl.Player(player_name=args.player)
    stream = WebcamVideoStream(args.stream_source)
    detector = FaceDetector(Path(args.classifier_params))
    Controller(player, stream, detector, Path(args.logdir)).start()

    glib_loop = GLib.MainLoop()
    glib_loop.run()
    return 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and executable module to control the player with your face."""

import json
import logging
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import gi
import tensorflow as tf
from gi.repository import GLib, Playerctl

from facectrl.detector import FaceDetector
from facectrl.video import Tracker, VideoStream


class Controller:
    """Controls everything that passes on the bus and on the camera."""

    def __init__(
        self,
        player_name,
        stream: VideoStream,
        detector: FaceDetector,
        logdir: Path,
        debug: bool,
    ):
        """The controller, that controles the player using the models."""
        self._desired_player = player_name
        self._manager = Playerctl.PlayerManager()
        self._manager.connect("name-appeared", self._on_name_appeared)
        self._manager.connect("name-vanished", self._on_name_vanished)
        self._player = None
        self._stream = stream
        self._detector = detector
        self._autoencoder = tf.keras.models.load_model(str(logdir / "on" / "saved"))

        with open(logdir / "on" / "best" / "LD" / "LD.json") as fp:
            json_file = json.load(fp)
            self._thresh = {
                "on": float(json_file["positive_threshold"]),
                "off": float(json_file["negative_threshold"]),
            }
        self._mse = tf.keras.losses.MeanSquaredError()
        self._debug = debug
        self._playing = False
        self._stop = False

        # Start main loop
        glib_loop = GLib.MainLoop()
        glib_loop.run()

    def _on_name_vanished(self, manager, name):
        self._stop = True

    def _on_name_appeared(self, manager, name):
        if name.name != self._desired_player:
            pass
        self._player = Playerctl.Player.new_from_name(name)

        self._player.connect("playback-status::playing", self._on_play)
        self._player.connect("playback-status::paused", self._on_pause)
        self._player.connect("playback-status::stopped", self._on_stop)

        self._manager.manage_player(self._player)
        self._start()

    def _on_play(self, player, status):
        logging.info("[PLAY] Setting to True")
        self._playing = True

    def _on_pause(self, player, status):
        logging.info("[PAUSE] Setting to False")
        self._playing = False

    def _on_stop(self, player, status):
        logging.info("[STOP] Setting to False")
        self._playing = False

    def _detect_and_classify(self, frame):
        bounding_box = self._detector.detect(frame)
        classified = ""

        if bounding_box[-1] != 0:
            logging.info("Face found!")

            # extract the face and classify it
            face = tf.expand_dims(
                tf.image.resize(
                    tf.image.convert_image_dtype(
                        FaceDetector.crop(frame, bounding_box, expansion=(70, 70)),
                        tf.float32,
                    ),
                    (64, 64),
                ),
                axis=[0],
            )

            reconstruction = self._autoencoder(face)
            mse = self._mse(face, reconstruction)
            logging.info("mse: %f", mse.numpy())
            if mse <= self._thresh["on"]:
                classified = "on"
            if mse >= self._thresh["off"]:
                classified = "off"

            if classified:
                logging.info("Classified as: %s with mse %f", classified, mse)
                if self._debug:
                    cv2.imshow(
                        "reconstruction",
                        tf.squeeze(
                            tf.image.convert_image_dtype(reconstruction, tf.uint8)
                        ).numpy(),
                    )
            else:
                logging.info(
                    "Unable to classify the input because mse %s is outside of positive %f and negative %f",
                    mse,
                    self._thresh["on"],
                    self._thresh["off"],
                )

        return classified, bounding_box

    def _start(self):
        with self._stream:
            while not self._stop:
                classified = ""
                # find the face for the first time in this loop
                logging.info("Detecting a face...")
                start = time.time()
                while not classified:
                    frame = self._stream.read()

                    # More than 2 seconds without a detected face: pause
                    if self._playing and time.time() - start > 2:
                        logging.info("Nobody in front of the camera (?)")
                        self._player.pause()

                    classified, bounding_box = self._detect_and_classify(frame)

                tracker = Tracker(
                    frame,
                    bounding_box,
                    max_failures=self._stream.fps,
                    debug=self._debug,
                )

                if not self._playing and classified == "on":
                    logging.info("PLAY")
                    self._player.play()
                if self._playing and classified == "off":
                    logging.info("PAUSE")
                    self._player.pause()
                try:
                    # Start tracking stream
                    while not self._stop:
                        frame = self._stream.read()
                        tracker.track(frame)
                except ValueError:
                    logging.info("Tracking failed")
                else:
                    logging.info("Unable to classify the input")

        # Get ready to restart
        self._playing = False
        self._stop = False
        if self._debug:
            cv2.destroyAllWindows()


def main():
    """Main method, invoked with python -m facectrl.ctrl."""
    gi.require_version("Playerctl", "2.0")
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--player", required=True, type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--classifier-params", required=True)
    parser.add_argument("--stream-source", default=0)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    stream = VideoStream(args.stream_source)
    detector = FaceDetector(Path(args.classifier_params))
    Controller(args.player, stream, detector, Path(args.logdir), args.debug)

    return 1


if __name__ == "__main__":
    sys.exit(main())

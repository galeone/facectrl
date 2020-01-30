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
from typing import Tuple

import cv2
import gi
import tensorflow as tf
from gi.repository import GLib, Playerctl

from facectrl.ml import ClassificationResult, Classifier, FaceDetector
from facectrl.video import Tracker, VideoStream


class Controller:
    """Controls everything that passes on the bus and on the camera.
    This is a Glib application"""

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
        self._debug = debug

        with open(logdir / "on" / "best" / "LD" / "LD.json") as fp:
            json_file = json.load(fp)
            thresholds = {
                "LD": float(json_file["LD"]),
                "on": float(json_file["positive_threshold"]),
                "on_variance": float(json_file["positive_variance"]),
                "off": float(json_file["negative_threshold"]),
                "off_variance": float(json_file["negative_variance"]),
            }

        self._classifier = Classifier(
            autoencoder=tf.keras.models.load_model(str(logdir / "on" / "saved")),
            thresholds=thresholds,
            debug=self._debug,
        )

        self._playing = False
        self._stop = False

        # Start main loop
        glib_loop = GLib.MainLoop()
        glib_loop.run()

    def _on_name_vanished(self, manager, name):
        self._stop = True
        self._playing = False

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

    def _detect_and_classify(self, frame) -> Tuple[ClassificationResult, Tuple]:
        bounding_box = self._detector.detect(frame)
        classification_result = ClassificationResult.UNKNOWN

        if bounding_box[-1] != 0:
            logging.info("Face found!")

            classification_result = self._classifier(
                Classifier.preprocess(
                    FaceDetector.crop(frame, bounding_box, expansion=(70, 70))
                )
            )

        return classification_result, bounding_box

    def _decide(self, classification_result):
        if self._stop:
            pass

        if (
            not self._playing
            and classification_result is ClassificationResult.HEADPHONES_ON
        ):
            logging.info("PLAY")
            self._player.play()
        if (
            self._playing
            and classification_result is ClassificationResult.HEADPHONES_OFF
        ):
            logging.info("PAUSE")
            self._player.pause()

    def _start(self):
        with self._stream:
            while not self._stop:
                classification_result = ClassificationResult.UNKNOWN
                # find the face for the first time in this loop
                logging.info("Detecting a face...")
                start = time.time()
                while classification_result is ClassificationResult.UNKNOWN:
                    frame = self._stream.read()

                    # More than 2 seconds without a detected face: pause
                    if self._playing and time.time() - start > 2 and not self._stop:
                        logging.info("Nobody in front of the camera (?)")
                        self._player.pause()

                    classification_result, bounding_box = self._detect_and_classify(
                        frame
                    )

                tracker = Tracker(
                    frame,
                    bounding_box,
                    max_failures=self._stream.fps,
                    debug=self._debug,
                )
                tracker.classifier = self._classifier

                self._decide(classification_result)

                try:
                    # Start tracking stream
                    while not self._stop:
                        frame = self._stream.read()
                        classification_result = tracker.track_and_classify(frame)
                        self._decide(classification_result)

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

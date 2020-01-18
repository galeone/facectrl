# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and executable module to control the player with your face."""

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
        self,
        player,
        stream: WebcamVideoStream,
        detector: FaceDetector,
        models_dir: Path,
    ):
        """The controller, that controles the player using the models."""
        self._player = player
        self._stream = stream
        self._detector = detector
        self._models = {
            "on": tf.keras.models.load_model(str(models_dir / "on")),
            "off": tf.keras.models.load_model(str(models_dir / "off")),
        }
        self._tracker = None

    def _find_face(self):
        with self._stream:
            while True:
                frame = self._stream.read()
                bounding_box = self._detector.detect(frame)
                if bounding_box[-1] != 0:
                    return frame, bounding_box

    def start(self):
        frame, bounding_box = self._find_face()
        face = FaceDetector.crop(frame, bounding_box, expansion=(30, 30))

        self._tracker = Tracker(frame, bounding_box)


def main():
    """Main method, invoked with python -m facectrl.ctrl."""
    gi.require_version("Playerctl", "2.0")
    from gi.repository import Playerctl, GLib  # pylint: disable=import-outside-toplevel

    parser = ArgumentParser()
    parser.add_argument("--player", required=True, type=str)
    parser.add_argument("--models_dir", required=True, type=str)
    args = parser.parse_args()

    player = Playerctl.Player(player_name=args.player)
    Controller(player, Path(args.models_dir)).start()

    glib_loop = GLib.MainLoop()
    glib_loop.run()


if __name__ == "__main__":
    sys.exit(main())

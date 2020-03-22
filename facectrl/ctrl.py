# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and executable module to control the player with your face."""

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from gi.repository import GLib, GObject, Playerctl

from facectrl.ml import (ClassificationResult, Classifier, FaceDetector,
                         Thresholds)
from facectrl.video import Tracker, VideoStream


class Controller:
    """The controller. It controls your player, tracks and classify your face.
    This is a Glib application"""

    def __init__(
        self,
        player_name,
        stream: VideoStream,
        detector: FaceDetector,
        logdir: Path,
        metric_name: str,
        debug: bool,
    ) -> None:
        """The controller. It controls your player, tracks and classify your face."""
        self._desired_player = player_name
        self._manager = Playerctl.PlayerManager()
        self._manager.connect("name-appeared", self._on_name_appeared)
        self._manager.connect("name-vanished", self._on_name_vanished)
        self._player = None
        self._stream = stream
        self._detector = detector
        self._debug = debug
        self._expansion = (70, 70)
        self._counters = {hit: 0 for hit in ClassificationResult}
        # FPS value of this application. It has nothing to the with the
        # VideoStram.fps value.
        self._fps = -1

        with open(
            logdir / "on" / "best" / metric_name / (metric_name + ".json")
        ) as json_fp:
            json_file = json.load(json_fp)
            thresholds = Thresholds(
                on={
                    "mean": float(json_file.get("positive_threshold", -1)),
                    "variance": float(json_file.get("positive_variance", -1)),
                },
                off={
                    "mean": float(json_file.get("negative_threshold", -1)),
                    "variance": float(json_file.get("negative_variance", -1)),
                },
            )

        model = tf.saved_model.load(str(logdir / "on" / "saved"))
        self._classifier = Classifier(
            model=model, thresholds=thresholds, debug=self._debug,
        )

        # These 3 bools control the player using the camera
        self._play = False
        self._stop = False
        self._pause = False
        # However, we need to let the user control the player
        # thus we set another flag that is set only if the play(),pause(),stop()
        # methods have been called from this application.
        self._internal = False

        # Start main loop
        glib_loop = GLib.MainLoop()
        glib_loop.run()

    def _fps_value(self):
        if self._fps < 0:
            # Here the best guess for a second
            # is the VideoStream FPS value.
            # However, this will last longer than one
            # second for sure, since the VideStream FPS value
            # doens't take into account the image processing
            # time.
            return self._stream.fps
        return self._fps

    def _reset(self):
        self._play = False
        self._stop = False
        self._pause = False
        self._player = None

    def _pause_player(self):
        self._internal = True
        if self._player:
            try:
                self._player.pause()
            except GLib.Error:
                self._reset()
        else:
            self._reset()

    def _play_player(self):
        self._internal = True
        if self._player:
            try:
                self._player.play()
            except GLib.Error:
                self._reset()
        else:
            self._reset()

    def _stop_player(self):
        self._internal = True
        if self._player:
            try:
                self._player.stop()
            except GLib.Error:
                self._reset()
        else:
            self._reset()

    def _on_name_vanished(self, manager, name) -> None:
        logging.info("Player vanising...")
        self._reset()

    def _on_name_appeared(self, manager, name) -> None:
        if name.name != self._desired_player:
            logging.info(
                "Appeared player: %s but expected %s", name.name, self._desired_player
            )
            return
        self._player = Playerctl.Player.new_from_name(name)
        self._player.connect("playback-status::playing", self._on_play)
        self._player.connect("playback-status::paused", self._on_pause)
        self._player.connect("playback-status::stopped", self._on_stop)
        self._manager.manage_player(self._player)
        if self._debug:
            self._start()
        else:
            Thread(target=self._start).start()

    def _on_play(self, player, status) -> None:
        logging.info("ON PLAY")
        if self._internal:
            self._internal = False
            self._play = True
            self._stop = False
            self._pause = False

    def _on_stop(self, player, status) -> None:
        logging.info("ON STOP")
        if self._internal:
            self._internal = False
            self._play = False
            self._stop = True
            self._pause = False

    def _on_pause(self, player, status) -> None:
        logging.info("ON PAUSE")
        if self._internal:
            self._internal = False

            self._play = False
            self._stop = False
            self._pause = True

    def _detect_and_classify(self, frame) -> Tuple[ClassificationResult, Tuple]:
        bounding_box = self._detector.detect(frame)
        classification_result = [ClassificationResult.UNKNOWN]

        if bounding_box[-1] != 0:
            logging.info("Face found!")

            classification_result = self._classifier(
                Classifier.preprocess(
                    FaceDetector.crop(frame, bounding_box, expansion=self._expansion)
                )
            )

        return classification_result[0], bounding_box

    def _decide(
        self,
        classification_result: ClassificationResult,
        previous_result: ClassificationResult,
    ) -> None:
        if classification_result == previous_result:
            self._counters[classification_result] += 1
        else:
            for hit in ClassificationResult:
                self._counters[hit] = 0

        # Stabilize the predictions: take a NEW action only if all the latest
        # predictions agree with this one
        if self._counters[classification_result] >= self._fps_value():
            if self._is_playing():
                logging.info("is playing")
                if classification_result in (
                    ClassificationResult.HEADPHONES_OFF,
                    ClassificationResult.UNKNOWN,
                ):
                    logging.info("decide: pause")
                    self._pause_player()
                    return
            elif self._is_stopped() or self._is_paused():
                logging.info("is stopped or paused")
                if classification_result == ClassificationResult.HEADPHONES_ON:
                    logging.info("decide: play")
                    self._play_player()
                    return

        logging.info("decide: dp nothing")

    def _is_stopped(self) -> bool:
        return self._stop

    def _is_playing(self) -> bool:
        return self._play

    def _is_paused(self) -> bool:
        return self._is_paused

    def _start(self) -> None:
        with self._stream:
            while not self._is_stopped() and self._player:
                classification_result = ClassificationResult.UNKNOWN
                # find the face for the first time in this loop
                logging.info("Detecting a face...")
                start = time()
                while (
                    classification_result == ClassificationResult.UNKNOWN
                    and self._player
                ):
                    frame = self._stream.read()

                    # More than 2 seconds without a detected face: pause
                    if (
                        self._is_playing()
                        and time() - start > 2
                        and not self._is_stopped()
                    ):
                        logging.info("Nobody in front of the camera")
                        self._pause_player()
                        sleep(1)

                    classification_result, bounding_box = self._detect_and_classify(
                        frame
                    )

                previous_result = classification_result
                self._decide(classification_result, previous_result)

                tracker = Tracker(
                    frame,
                    bounding_box,
                    # Halve the FPS value because at this time, self.fps
                    # Is the VideoStream FPS - without taking into account the
                    # image processing.
                    max_failures=self._fps_value() // 2,
                    debug=self._debug,
                )
                tracker.classifier = self._classifier

                try:
                    # Start tracking stream
                    frame_counter = 0
                    while not self._is_stopped() and self._player:
                        if self._fps < 0:
                            start = time()
                        frame = self._stream.read()
                        classification_result = tracker.track_and_classify(
                            frame, expansion=self._expansion
                        )
                        self._decide(classification_result, previous_result)
                        previous_result = classification_result
                        if self._fps < 0:
                            frame_counter += 1
                            if time() - start >= 1:
                                self._fps = frame_counter
                                # Update tracker max_failures
                                tracker.max_failures = self._fps_value()

                except ValueError:
                    logging.info("Tracking failed")
                    self._pause_player()

        # Get ready to restart
        if self._debug:
            cv2.destroyAllWindows()
        self._reset()


def main():
    """Main method, invoked with python -m facectrl.ctrl."""
    parser = ArgumentParser()
    parser.add_argument("--player", required=True, type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--classifier-params", required=True)
    parser.add_argument("--stream-source", default=0)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)

    stream = VideoStream(args.stream_source)
    detector = FaceDetector(Path(args.classifier_params))
    Controller(
        args.player, stream, detector, Path(args.logdir), args.metric, args.debug
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and utilities to use video streams."""

import cv2
import numpy as np
from facectrl.detector import FaceDetector
from facectrl.ml import ClassificationResult


class Tracker:
    """Tracks one object. It uses the CSRT tracker."""

    def __init__(
        self,
        frame,
        bounding_box,
        classifier,
        max_failures=10,
        name="face",
        debug: bool = False,
    ) -> None:
        """Initialize the frame tracker: start tracking the object
        localized into the bounding box in the current frame.
        Args:
            frame: BGR input image
            bounding_box: the bounding box containing the object to track
            max_failures: the number of frame to skip, before raising an
                          exception during the "track" call.
            name: an identifier for the object to track
            debug: set to true to enable visual debugging (opencv window)
        Returns:
            None
        """
        self._name = name
        self._tracker = cv2.TrackerCSRT_create()
        self._golden_crop = FaceDetector.crop(frame, tuple(bounding_box))
        self._tracker.init(frame, bounding_box)
        self._max_failures = max_failures
        self._failures = 0
        self._debug = debug
        self._classifier = classifier

    def track_and_classify(self, frame):
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
        """
        success, bounding_box = self._tracker.update(frame)
        classification_result = ClassificationResult.UNKNOWN
        if success:
            self._failures = 0
            if self._debug:
                bounding_box = np.array(bounding_box, dtype=np.int32)
                cv2.rectangle(
                    frame,
                    tuple(bounding_box[:2]),
                    tuple(bounding_box[:2] + bounding_box[2:]),
                    (0, 255, 0),
                )
                cv2.imshow("debug", frame)
                cv2.waitKey(1)

            crop = FaceDetector.crop(frame, bounding_box, expansion=(70, 70))
            classification_result = self._classifier(self._classifier.preprocess(crop))
        else:
            self._failures += 1
            if self._failures >= self._max_failures:
                if self._debug:
                    cv2.destroyAllWindows()
                raise ValueError(
                    f"Can't find {self._name} for {self._max_failures} times"
                )

        return classification_result

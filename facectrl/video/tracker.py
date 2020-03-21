# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and utilities to use video streams."""

from typing import Tuple

import cv2
import numpy as np

from facectrl.ml import ClassificationResult, Classifier, FaceDetector


class Tracker:
    """Tracks one object. It uses the CSRT tracker."""

    def __init__(
        self, frame, bounding_box, max_failures=10, debug: bool = False,
    ) -> None:
        """Initialize the frame tracker: start tracking the object
        localized into the bounding box in the current frame.
        Args:
            frame: BGR input image
            bounding_box: the bounding box containing the object to track
            max_failures: the number of frame to skip, before raising an
                          exception during the "track" call.
            debug: set to true to enable visual debugging (opencv window)
        Returns:
            None
        """
        self._tracker = cv2.TrackerCSRT_create()
        self._golden_crop = FaceDetector.crop(frame, tuple(bounding_box))
        self._tracker.init(frame, bounding_box)
        self._max_failures = max_failures
        self._failures = 0
        self._debug = debug
        self._classifier = None

    def track(self, frame) -> Tuple[bool, Tuple]:
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
        Returns:
            success, bounding_box: a boolean that indicates if the tracking succded
            and a bounding_box containing the tracked objecrt positon.
        """
        return self._tracker.update(frame)

    @property
    def classifier(self) -> Classifier:
        """Get the classifier previousluy set. None otherwise."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: Classifier) -> None:
        """
        Args:
            classifier: the Classifier to use
        """
        self._classifier = classifier

    @property
    def max_failures(self) -> int:
        """Get the max_failures value: the number of frame to skip
        before raising an exception during the "track" call."""
        return self._max_failures

    @max_failures.setter
    def max_failures(self, value):
        """Update the max_failures value."""
        self._max_failures = value

    def track_and_classify(
        self, frame: np.array, expansion=(100, 100)
    ) -> ClassificationResult:
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
            expansion: expand the ROI around the detected object by this amount
        Return:
            classification_result (ClassificationResult)
        """
        if not self._classifier:
            raise ValueError("You need to set a classifier first.")
        success, bounding_box = self.track(frame)
        classification_result = ClassificationResult.UNKNOWN
        if success:
            self._failures = 0

            if self._debug:
                bounding_box = np.array(bounding_box, dtype=np.int32)
                frame_copy = frame.copy()
                cv2.rectangle(
                    frame_copy,
                    tuple(bounding_box[:2]),
                    tuple(bounding_box[:2] + bounding_box[2:]),
                    (0, 255, 0),
                )
                cv2.imshow("debug", frame_copy)
                cv2.waitKey(1)

            crop = FaceDetector.crop(frame, bounding_box, expansion=expansion)
            classification_result = self._classifier(self._classifier.preprocess(crop))[
                0
            ]
        else:
            self._failures += 1
            if self._failures >= self._max_failures:
                if self._debug:
                    cv2.destroyAllWindows()
                raise ValueError(f"Can't find object for {self._max_failures} times")

        return classification_result

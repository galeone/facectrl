# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and utilities to use video streams."""

from threading import Lock, Thread

import cv2
import numpy as np


class WebcamVideoStream:
    """Non locking version of a VideoCapture stream."""

    def __init__(self, src: int = 0) -> None:
        """Initializes the WebcamVideoStream.
        Acquires the control of the VideoSource src and
        prevents concurrent accesses.

        Args:
            src: the ID of the capturing device.
        Returns:
            None
        """
        self.stream = cv2.VideoCapture(src)
        (_, self._frame) = self.stream.read()
        self._grabbing = False
        self._lock = Lock()
        self._thread = None

    # Below we use a forward declaration to correctly type annotating
    # the start method, that returns a type not yet conpletely defined
    def _start(self) -> "WebcamVideoStream":
        """Start the video stream. Creates a thread that initializes the
        self._frame variable with the first frame.
        Does nothing if the stream has already been started.

        Returns:
            self
        """
        if self._grabbing:
            # already started, do nothing
            return self
        self._grabbing = True
        self._thread = Thread(target=self._update, args=())
        self._thread.start()
        return self

    def _stop(self) -> None:
        """Stop the acquisition stream.
        Waits for the acquistion thread to terminate.
        """

        self._grabbing = False
        self._thread.join()
        # self.stream.release()

    def _update(self) -> None:
        """The update operation: updetes the self._frame variable.
        Thread safe.
        """
        while self._grabbing:
            (_, frame) = self.stream.read()
            self._lock.acquire()
            self._frame = frame
            self._lock.release()

    def read(self) -> np.array:
        """Returns the current grabbed frame.

        Returns:
            frame (np.array) the BGR image acquired.
        """
        self._lock.acquire()
        frame = self._frame.copy()
        self._lock.release()
        return frame

    def __enter__(self) -> "WebcamVideoStream":  # forward declaration
        """Start the WebcamVideoStream."""
        return self._start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the videostream, waits for all the thread to finis
        and clean up the resources.
        """
        self._stop()

    def __del__(self):
        """On object destruction, release the videostream."""
        self.stream.release()


class Tracker:
    """Tracks one object. It uses the MOSSE tracker."""

    def __init__(self, frame, bounding_box, max_failures=10, name="face") -> None:
        """Initialize the frame tracker: start tracking the object
        localized into the bounding box in the current frame.
        Args:
            frame: BGR input image
            bounding_box: the bounding box containing the object to track
            max_failures: the number of frame to skip, before raising an
                          exception during the "track" call.
            name: an identifier for the object to track
        Returns:
            None
        """
        self._name = name
        self._tracker = cv2.TrackerMOSSE_create()
        self._tracker.init(frame, bounding_box)
        self._max_failures = max_failures
        self._failures = 0

    def track(self, frame):
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
        """
        success, bounding_box = self._tracker.update(frame)
        if success:
            self._failures = 0
        else:
            self._failures += 1
            if self._failures >= self._max_failures:
                self._failures = 0
                raise ValueError(
                    f"Can't find {self._name} for {self._max_failures} times"
                )

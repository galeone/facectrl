# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and utilities to use video streams."""

from threading import Lock, Thread

import cv2
import numpy as np


class VideoStream:
    """Non locking version of a VideoCapture stream."""

    def __init__(self, src: int = 0) -> None:
        """Initializes the VideoStream.
        Acquires the control of the VideoSource src and
        prevents concurrent accesses.

        Args:
            src: the ID of the capturing device.
        Returns:
            None
        """
        self._src = src
        self._frame = np.array([])
        self._fps = 0
        self._stream = None
        self._grabbing = False
        self._lock = Lock()
        self._thread = None

    # Below we use a forward declaration to correctly type annotating
    # the start method, that returns a type not yet conpletely defined
    def _start(self) -> "VideoStream":
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

    def _update(self) -> None:
        """The update operation: updetes the self._frame variable.
        Thread safe.
        """
        while self._grabbing:
            (_, frame) = self._stream.read()
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

    @property
    def fps(self) -> float:
        """Return the current FPS value of the video stream.
        Please note that this is not the FPS of your application
        using this VideoStream.
        """
        if not self._fps:
            self._fps = self._stream.get(cv2.CAP_PROP_FPS)
        return self._fps

    def __enter__(self) -> "VideoStream":  # forward declaration
        """Acquire resources and start the VideoStream."""
        self._stream = cv2.VideoCapture(self._src)
        (_, self._frame) = self._stream.read()

        return self._start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the videostream, waits for all the thread to finis
        and clean up the resources.
        """
        self._stop()
        if self._stream:
            self._stream.release()

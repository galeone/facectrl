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

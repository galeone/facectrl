# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and utilities to detect faces into images."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class FaceDetector:
    """Initialize a classifier and uses it to detect the bigger
    face present into an image.
    """

    def __init__(self, params: Path) -> None:
        """Initializes the face detector using the specified parameters.

        Args:
            params: the path of the haar cascade classifier XML file.
        """
        self.classifier = cv2.CascadeClassifier(str(params))

    def detect(self, frame: np.array) -> Tuple:
        """Search for faces into the input frame.
        Returns the bounding box containing the bigger (close to the camera)
        detected face, if any.
        When no face is detected, the tuple returned has width and height void.

        Args:
            frame: the BGR input image.
        Returns:
            (x,y,w,h): the bounding box.
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proposals = np.array(
            self.classifier.detectMultiScale(
                frame,
                scaleFactor=1.5,  # 50%
                # that's big, but we're interested
                # in detecting faces closes to the camera, so
                # this is OK.
                minNeighbors=4,
                # We want at least "minNeighbors" detections
                # around the same face,
                minSize=(frame.shape[0] // 4, frame.shape[1] // 4),
                # Only bigger faces -> we suppose the face to be at least
                # 25% of the content of the input image
                maxSize=(frame.shape[0], frame.shape[1]),
            )
        )

        # If faces have been detected, find the bigger one
        if proposals.size:
            bigger_id = 0
            bigger_area = 0
            for idx, (_, _, width, height) in enumerate(proposals):
                area = width * height
                if area > bigger_area:
                    bigger_id = idx
                    bigger_area = area
            return tuple(proposals[bigger_id])  # (x,y,w,h)
        return (0, 0, 0, 0)

    @staticmethod
    def crop(frame, bounding_box, expansion=(0, 0)) -> np.array:
        """
        Extract from the input frame the content of the bounding_box.
        Applies the required expension to the bounding box.

        Args:
            frame: BGR image
            bounding_box: tuple with format (x,y,w,h)
            expansion: the amount of pixesl the add to increase the
                       bouding box size, from the center.
        Returns:
            cropped: BGR image with size, at least (bounding_box[2], bounding_box[3]).
        """

        x, y, width, height = [
            int(element) for element in bounding_box
        ]  # pylint: disable=invalid-name

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        halfs = (expansion[0] // 2, expansion[1] // 2)
        if width + halfs[0] <= frame.shape[1]:
            width += halfs[0]
        if x - halfs[0] >= 0:
            x -= halfs[0]
        if height + halfs[1] <= frame.shape[0]:
            height += halfs[1]
        if y - halfs[1] >= 0:
            y -= halfs[1]

        image_crop = frame[y : y + height, x : x + width]
        return image_crop

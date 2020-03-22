# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Classes and executable module to use to create the datasets."""

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from facectrl.ml import FaceDetector
from facectrl.video import Tracker, VideoStream


class Builder:
    """Builds the dataset interactively."""

    def __init__(self, dest: Path, params: Path, src: int = 0) -> None:
        """Initializes the dataset builder.

        Args:
            dest: the destination folder for the dataset.
            params: Path of the haar cascade classifier parameters.
            src: the ID of the video stream to use (input of VideoStream).
        Returns:
            None
        """
        self._on_dir = dest / "on"
        self._off_dir = dest / "off"
        if not self._on_dir.exists():
            os.makedirs(self._on_dir)
        if not self._off_dir.exists():
            os.makedirs(self._off_dir)
        self._stream = VideoStream(src)
        self._detector = FaceDetector(params)

    def _acquire(self, path, expansion, prefix) -> None:
        """Acquire and store into path the samples.
        Args:
            path: the path where to store the cropped images.
            expansion: the expansion to apply to the bounding box detected.
            prefix: prefix added to the opencv window
        Returns:
            None
        """
        i = 0
        quit_key = ord("q")
        start = len(list(path.glob("*.png")))
        with self._stream:
            detected = False
            while not detected:
                frame = self._stream.read()
                bounding_box = self._detector.detect(frame)
                detected = bounding_box[-1] != 0

            tracker = Tracker(frame, bounding_box)
            success = True
            while success:
                success, bounding_box = tracker.track(frame)
                if success:
                    bounding_box = np.int32(bounding_box)
                    crop = FaceDetector.crop(frame, bounding_box, expansion)
                    crop_copy = crop.copy()
                    cv2.putText(
                        crop_copy,
                        f"{prefix} {i + 1}",
                        (30, crop.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                    )
                    cv2.imshow("grab", crop_copy)
                    key = cv2.waitKey(1) & 0xFF
                    if key == quit_key:
                        success = False
                    cv2.imwrite(str(path / Path(str(start + i) + ".png")), crop)
                    i += 1
                frame = self._stream.read()
        cv2.destroyAllWindows()

    def headphones_on(self, expansion=(70, 70)) -> None:
        """Acquire and store the images with the headphones on.
        Args:
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._on_dir, expansion, "ON")

    def headphones_off(self, expansion=(70, 70)) -> None:
        """Acquire and store the images with the headphones off.
        Args:
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._off_dir, expansion, "OFF")


def main() -> int:
    """Main method, invoked with python -m facectrl.dataset."""
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-params", required=True)
    parser.add_argument("--stream-source", default=0)
    parser.add_argument("--expansion", default=70)

    args = parser.parse_args()

    builder = Builder(
        Path(args.dataset_path), Path(args.classifier_params), args.stream_source
    )
    print(
        "Acquiring pictures WITH HEADPHONES ON in 5 seconds..."
        "Press [Q] to stop capturing."
    )
    time.sleep(5)
    builder.headphones_on((args.expansion, args.expansion))
    print(
        "Acquiring picture WITH HEADPHONES OFF in 5 seconds..."
        "Press [Q] to stop capturing."
    )
    time.sleep(5)
    builder.headphones_off((args.expansion, args.expansion))
    return 0


if __name__ == "__main__":
    sys.exit(main())

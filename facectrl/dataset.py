"""Classes and executable module to use to create the datasets."""
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import tensorflow as tf

from facectrl.detector import FaceDetector
from facectrl.video import WebcamVideoStream


class Builder:
    """Builds the dataset interactively."""

    def __init__(self, dest: Path, params: Path, src: int = 0) -> None:
        """Initializes the dataset builder.

        Args:
            dest: the destination folder for the dataset.
            params: Path of the haar cascade classifier parameters.
            src: the ID of the video stream to use (input of WebcamVideoStream).
        Returns:
            None
        """
        self._on_dir = dest / "on"
        self._off_dir = dest / "off"
        if not self._on_dir.exists():
            os.makedirs(self._on_dir)
        if not self._off_dir.exists():
            os.makedirs(self._off_dir)
        self._stream = WebcamVideoStream(src)
        self._detector = FaceDetector(params)

    def _acquire(self, path, num_samples, expansion, prefix) -> None:
        """Acquire and store into path the samples.
        Args:
            path: the path where to store the cropped images.
            num_samples: the number of samples to save before exiting.
            expansion: the expansion to apply to the bounding box detected.
            prefix: prefix added to the opencv window
        Returns:
            None
        """
        i = 0
        yes_no_keys = [ord("y"), ord("n")]
        with self._stream:
            while i < num_samples:
                frame = self._stream.read()
                bounding_box = self._detector.detect(frame)
                if bounding_box[-1] != 0:
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
                    cv2.imshow(f"{prefix} Accept [y/n]", crop_copy)
                    key = cv2.waitKey(0) & 0xFF
                    while key not in yes_no_keys:
                        print("Accept with Y, reject with N")
                        key = cv2.waitKey(0) & 0xFF
                    if key == yes_no_keys[0]:
                        cv2.imwrite(str(path / Path(str(i) + ".png")), crop)
                        i += 1
        cv2.destroyAllWindows()

    def on(self, num_samples=50, expansion=(30, 30)):
        """Acquire and store the images with the headphones on.
        Args:
            num_samples: the number of samples to save before exiting.
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._on_dir, num_samples, expansion, "ON")

    def off(self, num_samples=50, expansion=(30, 30)):
        """Acquire and store the images with the headphones off.
        Args:
            num_samples: the number of samples to save before exiting.
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._off_dir, num_samples, expansion, "OFF")


def main():
    """Main method, invoked with python -m facectrl.dataset."""
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-params", required=True)
    parser.add_argument("--stream-source", default=0)
    parser.add_argument("--num-samples", default=50, type=int)
    parser.add_argument("--expansion", default=30)

    args = parser.parse_args()

    builder = Builder(
        Path(args.dataset_path), Path(args.classifier_params), args.stream_source
    )
    print("Acquiring picture WITH HEADPHONES ON in 5 seconds...")
    time.sleep(5)
    builder.on(args.num_samples, (args.expansion, args.expansion))
    print("Acquiring picture WITH HEADPHONES OFF in 5 seconds...")
    time.sleep(5)
    builder.off(args.num_samples, (args.expansion, args.expansion))


if __name__ == "__main__":
    sys.exit(main())

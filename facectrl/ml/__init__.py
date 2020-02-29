# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Machine learning models used to detect if you're wearning headphones.
The package contains everything needed: models, training loop and dataset creation.
"""

from facectrl.ml.classifier import ClassificationResult, Classifier, Thresholds
from facectrl.ml.detector import FaceDetector
from facectrl.ml.train import AEAccuracy, MaximizeELBO, ReconstructionLoss, train

__ALL__ = [
    "ReconstructionLoss",
    "AEAccuracy",
    "MaximizeELBO",
    "train",
    "ClassificationResult",
    "Classifier",
    "Thresholds",
]

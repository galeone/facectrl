# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FaceCTRL package."""

from better_setuptools_git_version import get_version

from facectrl import ctrl, dataset, ml, video

__version__ = get_version()
__url__ = "https://github.com/galeone/facectrl"
__author__ = "Paolo Galeone"
__email__ = "me@pgaleone.eu"

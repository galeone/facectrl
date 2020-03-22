# Copyright 2020 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FaceCTRL: control your media player with your face."""

import re

from setuptools import find_packages, setup

if __name__ == "__main__":
    with open("README.md", "r", encoding="utf-8") as FP:
        README = FP.read()

    with open("requirements.txt") as FP:
        REQUIREMENTS = FP.read().splitlines()

    # Meta
    INIT_PY = open("facectrl/__init__.py").read()
    METADATA = dict(re.findall(r"__([a-z]+)__ = \"([^\"]+)\"", INIT_PY))

    setup(
        author=METADATA["author"],
        author_email=METADATA["email"],
        description="FaceCTRL: control your media player with your face.",
        install_requires=REQUIREMENTS,
        python_requires=">=3.7",
        license="Mozilla Public License Version 2.0",
        long_description=README,
        long_description_content_type="text/markdown",
        include_package_data=True,
        name="facectrl",
        packages=find_packages(),
        setup_requires=["better-setuptools-git-version"],
        tests_require=REQUIREMENTS,
        url=METADATA["url"],
        version_config={
            "version_format": "{tag}.dev{sha}",
            "starting_version": "0.0.1",
        },
    )

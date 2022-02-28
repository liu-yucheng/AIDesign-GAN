"""Package setup executable.

To be called by a package manager (pip or conda or others).
NOT supposed to be executed directly (via python or py).
Tells the package manager the way to install the source directory as a package.
The "entry_points" parameter of the setup function specifies the function to call when the user enters the
    corresponding command via the command line.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import setuptools
import shutil
from os import path as ospath

_copytree = shutil.copytree
_exists = ospath.exists
_find_packages = setuptools.find_packages
_setup = setuptools.setup


def _ensure_app_data():
    from aidesign_gan.libs import defaults

    path = defaults.app_data_path

    if _exists(path):
        print(f"Ensured app data at: {path}")
    else:
        default_path = defaults.default_app_data_path
        _copytree(default_path, path, dirs_exist_ok=True)
        print(f"Created app data at: {path}")


def main():
    _setup(
        name="aidesign-gan",
        version="0.71.0",
        description="AIDesign GAN Modeling Application",
        author="Yucheng Liu (From The AIDesign Team)",
        packages=_find_packages(),
        entry_points={
            "console_scripts": [
                "gan = aidesign_gan.exes.gan:main"
            ]
        }  # ,
        # test_suite="aidesign_gan.tests"
    )

    _ensure_app_data()

    # Check main command availability
    from aidesign_gan.exes import gan as _
    print("Commands available: gan")


if __name__ == "__main__":
    main()

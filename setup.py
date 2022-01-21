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

# Aliases

_find_packages = setuptools.find_packages
_setup = setuptools.setup

# End of aliases


def _make_default_app_data():
    from aidesign_gan.libs import defaults
    from aidesign_gan.libs import statuses
    from aidesign_gan.libs import utils

    # Aliases

    _init_folder = utils.init_folder
    _TrainStatus = statuses.GANTrainStatus
    _GenStatus = statuses.GANGenerateStatus

    # End of aliases

    _init_folder(defaults.app_data_path)

    gan_train_status = _TrainStatus()
    gan_generate_status = _GenStatus()

    gan_train_status.load()
    gan_generate_status.load()

    print(f"Ensured app data at: {defaults.app_data_path}")


def main():
    _setup(
        name="aidesign-gan",
        version="0.59.14",
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
    _make_default_app_data()
    print("Commands available: gan")

# Top level code


if __name__ == "__main__":
    main()

# End of top level code

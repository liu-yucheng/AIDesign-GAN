"""Package setup executable module.

When called by a package manager (pip/conda/python), this executable informs the package manager about how to install
the source directory as a package. The "entry_points" parameter of the setup function specifies the function to call
when the user enters the corresponding command to the command line.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import setuptools


def _make_default_app_data():
    from aidesign_gan.libs import defaults
    from aidesign_gan.libs import statuses
    from aidesign_gan.libs import utils

    utils.init_folder(defaults.app_data_path)
    gan_train_status = statuses.GANTrainStatus()
    gan_train_status.load()
    gan_generate_status = statuses.GANGenerateStatus()
    gan_generate_status.load()

    print(f"Created app data at: {defaults.app_data_path}")


def main():
    setuptools.setup(
        name="aidesign-gan",
        version="0.23.0",
        description="AIDesign GAN Modeling Application",
        author="The AIDesign Team",
        packages=setuptools.find_packages(),
        entry_points={
            "console_scripts": [
                "gan = aidesign_gan.exes.gan:main"
            ]
        }
        # test_suite="tests"
    )
    _make_default_app_data()
    print("Commands available: gan")


if __name__ == "__main__":
    main()

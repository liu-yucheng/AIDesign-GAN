"""Package setup script.

Inform the package manager (pip/conda/python) on how to install the source in the directory as a package. The
entry_points param of the setup function specifies the function to be called when the user enters the corresponding
command.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import setuptools


def _setup_exes_configs():
    from aidesign_dcgan.libs import configs
    train_config = configs.TrainConfig()
    train_config.load()
    generate_config = configs.GenerateConfig()
    generate_config.load()


def main():
    setuptools.setup(
        name="aidesign-dcgan",
        version="0.12.4",
        description="AI Design DCGAN Application",
        author="AI Design Team",
        packages=setuptools.find_packages(),
        entry_points={
            "console_scripts": [
                "dcgan-train = aidesign_dcgan.exes.train:main",
                "dcgan-generate = aidesign_dcgan.exes.generate:main"
            ]
        }
        # test_suite="tests"
    )
    _setup_exes_configs()


if __name__ == "__main__":
    main()

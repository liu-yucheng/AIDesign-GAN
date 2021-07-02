"""Package setup script.

Inform the package manager (pip/conda/python) on how to install the source in the directory as a package. The
entry_points param of the setup function specifies the function to be called when the user enters the corresponding
command.
"""

import setuptools

setuptools.setup(
    name="gan",
    version="0.5.0",
    description="AI Design GAN modeling application.",
    author="AI Design Team",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "gan-train = gan.exes.train:main",
            "gan-generate = gan.exes.generate:main"
        ],
    },
    # test_suite="tests"
)

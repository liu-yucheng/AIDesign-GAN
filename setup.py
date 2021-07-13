"""Package setup script.

Inform the package manager (pip/conda/python) on how to install the source in the directory as a package. The
entry_points param of the setup function specifies the function to be called when the user enters the corresponding
command.
"""

import setuptools

setuptools.setup(
    name="aidesign-dcgan",
    version="0.6.1",
    description="AI Design DCGAN Application",
    author="AI Design Team",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "dcgan-train = dcgan.exes.train:main",
            "dcgan-generate = dcgan.exes.generate:main"
        ]
    }
    # test_suite="tests"
)

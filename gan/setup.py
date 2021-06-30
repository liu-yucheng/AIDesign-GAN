"""Informs the package manager (pip/conda/python) on how to install the files
in the current directory as a package.

Note: the entry_points param of the setup function specifies the function to be
called when the user enters the corresponding command.
"""

import setuptools as setups

setups.setup(
    name='gan',
    version='0.4.0',
    description='AI Design GAN modeling application.',
    author='AI Design Team',
    packages=setups.tfind_packages(),
    entry_points={
        'console_scripts': [
            'gan-train = gan_exes.train:main',
            'gan-generate = gan_exes.generate:main'
        ],
    },
    # test_suite='tests'
)

"""Runs all the tests in the aidesign_gan.tests module."""

# First added by liu-yucheng
# Last updated by liu-yucheng

import pathlib
import unittest

from os import path

_join = path.join
_Path = pathlib.Path
_TestLoader = unittest.TestLoader
_TextTestRunner = unittest.TextTestRunner


_aidesign_gan_repo_path = str(_Path(__file__).parent)
_aidesign_gan_path = _join(_aidesign_gan_repo_path, "aidesign_gan")
_aidesign_gan_tests_path = _join(_aidesign_gan_path, "tests")


def main():
    """Runs this module as an executable."""
    loader = _TestLoader()
    suite = loader.discover(start_dir=_aidesign_gan_tests_path, pattern="test*.py")
    runner = _TextTestRunner(verbosity=1)
    runner.run(test=suite)


if __name__ == "__main__":
    main()

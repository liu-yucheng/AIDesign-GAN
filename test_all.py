"""Runs all the tests in the aidesign_gan.tests module."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import unittest

_TestLoader = unittest.TestLoader
_TextTestRunner = unittest.TextTestRunner


def main():
    """Runs this module as an executable."""
    loader = _TestLoader()
    suite = loader.discover(start_dir="aidesign_gan.tests", pattern="test*.py")
    runner = _TextTestRunner(verbosity=1)
    runner.run(test=suite)


if __name__ == "__main__":
    main()

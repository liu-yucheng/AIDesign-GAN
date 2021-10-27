"""Executable module for the "gan reset" command."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import sys

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Private attributes ...

_brief_usage = "gan reset"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

# ... Private attributes
# Nominal info strings ...

info = r"""Completed resetting the app data at: {}
"""
"""The primary info to display."""

# ... Nominal info strings
# Error info strings ...

too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 0 arguments; Gets {} arguments"""fr"""
{_usage}
"""
"""The info to display when the executable gets too many arguments."""

# ... Error info strings
# Other public attributes ...

argv_copy = None
"""A consumable copy of sys.argv."""

# ... Other public attributes


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length == 0:
        utils.init_folder(defaults.app_data_path, clean=True)

        gan_train_status = statuses.GANTrainStatus()
        gan_generate_status = statuses.GANGenerateStatus()
        gan_train_status.load()
        gan_generate_status.load()

        print(info.format(defaults.app_data_path), end="")
        exit(0)
    # elif argv_copy_length > 0
    else:
        print(too_many_args_info.format(argv_copy_length), end="")
        exit(1)


def main():
    """Starts the executable."""
    global argv_copy
    argv_length = len(sys.argv)
    assert argv_length >= 1
    argv_copy = copy.deepcopy(sys.argv)
    argv_copy.pop(0)
    run()
    exit(0)


# Let main be the script entry point
if __name__ == "__main__":
    main()

""""gan reset" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
# GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import sys

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_GenStatus = statuses.GANGenerateStatus
_init_folder = utils.init_folder
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan reset"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info strings

info = fr"""

Completed resetting the app data at: {{}}

"""
"""Primary info to display."""
info = info.strip()

# End of nominal info strings
# Error info strings

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 0 arguments; Gets {{}} arguments
{usage}

"""
"""Info to display when getting too many arguments."""
too_many_args_info = too_many_args_info.strip()

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        _init_folder(defaults.app_data_path, clean=True)

        gan_train_status = _TrainStatus()
        gan_generate_status = _GenStatus()
        gan_train_status.load()
        gan_generate_status.load()

        print(info.format(defaults.app_data_path))
        exit(0)
    else:  # elif argv_copy_length > 0:
        print(too_many_args_info.format(argv_copy_length), file=_stderr)
        exit(1)


def main():
    """Starts the executable."""
    global argv_copy
    argv_length = len(_argv)

    assert argv_length >= 1

    argv_copy = _deepcopy(_argv)
    argv_copy.pop(0)
    run()

# Top level code


if __name__ == "__main__":
    main()

# End of top level code

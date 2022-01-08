""""gan status" command executable.

Child command of "gan."
Can be launched directly.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import sys

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exists = ospath.exists
_GenStatus = statuses.GANGenerateStatus
_init_folder = utils.init_folder
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan status"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info strings

info = fr"""

App data is at: {{}}
"gan train" status:
{{}}
"gan generate" status:
{{}}

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
        if not _exists(defaults.app_data_path):
            _init_folder(defaults.app_data_path)

        app_data_info = defaults.app_data_path
        gan_train_info = ""
        gan_generate_info = ""

        gan_train_status = _TrainStatus()
        gan_generate_status = _GenStatus()
        gan_train_status.load()
        gan_generate_status.load()

        tab_width1 = 4
        tab_width2 = 8
        tab1 = " " * tab_width1
        gan_train_lines = []
        gan_generate_lines = []

        for key in gan_train_status.items:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = gan_train_status[key]
            line = f"{tab1}{key}:{tab2}{val}"
            gan_train_lines.append(line)

        for key in gan_generate_status.items:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = gan_generate_status[key]
            line = f"{tab1}{key}:{tab2}{val}"
            gan_generate_lines.append(line)

        gan_train_info = "\n".join(gan_train_lines)
        gan_generate_info = "\n".join(gan_generate_lines)

        print(info.format(app_data_info, gan_train_info, gan_generate_info))
        exit(0)
    else:  # elif argv_copy_length > 0:
        print(too_many_args_info.format(argv_copy_length), file=_stderr)
        exit(1)
    # end if


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

""""gan status" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import shutil
import sys

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses

# Aliases

_argv = sys.argv
_copytree = shutil.copytree
_deepcopy = copy.deepcopy
_exists = ospath.exists
_GenStatus = statuses.GANGenerateStatus
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
            _copytree(defaults.default_app_data_path, defaults.app_data_path, dirs_exist_ok=True)

        app_data_info = defaults.app_data_path
        train_info = ""
        gen_info = ""

        train_status = _TrainStatus.load_from_path(defaults.app_data_path)
        gen_status = _GenStatus.load_from_path(defaults.app_data_path)

        tab_width1 = 4
        tab_width2 = 8
        tab1 = " " * tab_width1
        train_lines = []
        gen_lines = []

        for key in train_status:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = train_status[key]
            line = f"{tab1}{key}:{tab2}{val}"
            train_lines.append(line)
        # end for

        for key in gen_status:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = gen_status[key]
            line = f"{tab1}{key}:{tab2}{val}"
            gen_lines.append(line)
        # end for

        train_info = "\n".join(train_lines)
        gen_info = "\n".join(gen_lines)

        print(info.format(app_data_info, train_info, gen_info))
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


if __name__ == "__main__":
    main()

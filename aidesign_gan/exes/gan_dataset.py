""""gan dataset ..." command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import pathlib
import sys

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exists = ospath.exists
_exit = sys.exit
_isabs = ospath.isabs
_isdir = ospath.isdir
_join = ospath.join
_Path = pathlib.Path
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan dataset <path-to-dataset>"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

Selected the dataset at: {{}}
Applied the selection to "gan train"

""".strip()
"""Primary info to display."""

# End of nominal info strings
# Error info strings

too_few_args_info = fr"""

"{brief_usage}" gets too few arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too few arguments."""

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too many arguments."""

dataset_does_not_exist_info = fr"""

"{brief_usage}" cannot find the dataset
Please check if the dataset is present at: {{}}
{usage}

""".strip()
"""Info to display when the selected dataset does not exist."""

dataset_is_not_dir_info = fr"""

"{brief_usage}" finds that the dataset is not a directory
Please check if the dataset appears as a directory at: {{}}
{usage}

""".strip()
"""Info to display when the selected dataset is not a directory."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)

    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), file=_stderr)
        _exit(1)
    elif argv_copy_length == 1:
        assert argv_copy is not None
        path_to_dataset = str(argv_copy.pop(0))

        if not _isabs(path_to_dataset):
            path_to_dataset = _join(".", path_to_dataset)

        path_to_dataset = str(_Path(path_to_dataset).resolve())

        if not _exists(path_to_dataset):
            print(dataset_does_not_exist_info.format(path_to_dataset), file=_stderr)
            _exit(1)

        if not _isdir(path_to_dataset):
            print(dataset_is_not_dir_info.format(path_to_dataset), file=_stderr)
            _exit(1)

        status = _TrainStatus.load_from_path(defaults.app_data_path)
        status["dataset_path"] = path_to_dataset
        status = _TrainStatus.verify(status)
        _TrainStatus.save_to_path(status, defaults.app_data_path)

        print(info.format(path_to_dataset))
        _exit(0)
    else:  # elif argv_copy_length > 1:
        print(too_many_args_info.format(argv_copy_length), file=_stderr)
        _exit(1)
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

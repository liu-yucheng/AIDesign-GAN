""""gan model ..." command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
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
_ExportStatus = statuses.GANExportStatus
_GenStatus = statuses.GANGenerateStatus
_isabs = ospath.isabs
_isdir = ospath.isdir
_join = ospath.join
_Path = pathlib.Path
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan model <path-to-model>"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info
info = fr"""

Selected the model at: {{}}
Applied the selection to the following commands:
    "gan train", "gan generate", "gan export ..."

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

model_does_not_exist_info = fr"""

"{brief_usage}" cannot find the model
Please check if the model is present at: {{}}
{usage}

""".strip()
"""Info to display when the selected model does not exist."""

model_is_not_dir_info = fr"""

"{brief_usage}" finds that the model is not a directory
Please check if the model appears as a directory at: {{}}
{usage}

""".strip()
"""Info to display when the selected model is not a directory."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)

    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), file=_stderr)
        exit(1)
    elif argv_copy_length == 1:
        assert argv_copy is not None
        path_to_model = argv_copy.pop(0)
        path_to_model = str(path_to_model)

        if not _isabs(path_to_model):
            path_to_model = _join(".", path_to_model)

        path_to_model = str(_Path(path_to_model).resolve())

        if not _exists(path_to_model):
            print(model_does_not_exist_info.format(path_to_model), file=_stderr)
            exit(1)

        if not _isdir(path_to_model):
            print(model_is_not_dir_info.format(path_to_model), file=_stderr)
            exit(1)

        train_status = _TrainStatus.load_from_path(defaults.app_data_path)
        gen_status = _GenStatus.load_from_path(defaults.app_data_path)
        export_status = _ExportStatus.load_from_path(defaults.app_data_path)

        train_status["model_path"] = path_to_model
        gen_status["model_path"] = path_to_model
        export_status["model_path"] = path_to_model

        train_status = _TrainStatus.verify(train_status)
        gen_status = _GenStatus.verify(gen_status)
        export_status = _ExportStatus.verify(export_status)

        _TrainStatus.save_to_path(train_status, defaults.app_data_path)
        _GenStatus.save_to_path(gen_status, defaults.app_data_path)
        _ExportStatus.save_to_path(export_status, defaults.app_data_path)

        print(info.format(path_to_model))
        exit(0)
    else:  # elif argv_copy_length > 1:
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

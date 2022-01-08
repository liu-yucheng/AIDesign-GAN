""""gan model" command executable.

Child command of "gan."
Can be launched directly.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import pathlib
import sys

from os import path as ospath

from aidesign_gan.libs import statuses

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exists = ospath.exists
_GenStatus = statuses.GANGenerateStatus
_isdir = ospath.isdir
_Path = pathlib.Path
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan model <path-to-model>"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info
info = fr"""

Selected the model at: {{}}
Applied the selection to "gan train" and "gan generate"

"""
"""Primary info to display."""
info = info.strip()

# End of nominal info strings
# Error info strings

too_few_args_info = fr"""

"{brief_usage}" gets too few arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

"""
"""Info to display when getting too few arguments."""
too_few_args_info = too_few_args_info.strip()

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

"""
"""Info to display when getting too many arguments."""
too_many_args_info = too_many_args_info.strip()

model_does_not_exist_info = fr"""

"{brief_usage}" cannot find the model
Please check if the model is present at: {{}}
{usage}

"""
"""Info to display when the selected model does not exist."""
model_does_not_exist_info = model_does_not_exist_info.strip()

model_is_not_dir_info = fr"""

"{brief_usage}" finds that the model is not a directory
Please check if the model appears as a directory at: {{}}
{usage}

"""
"""Info to display when the selected model is not a directory."""
model_is_not_dir_info = model_is_not_dir_info.strip()

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
        path_to_model = argv_copy.pop(0)
        path_to_model = str(path_to_model)
        path_to_model = "./" + path_to_model
        path_to_model = str(_Path(path_to_model).resolve())

        if not _exists(path_to_model):
            print(model_does_not_exist_info.format(path_to_model), file=_stderr)
            exit(1)
        if not _isdir(path_to_model):
            print(model_is_not_dir_info.format(path_to_model), file=_stderr)
            exit(1)

        gan_train_status = _TrainStatus()
        gan_generate_status = _GenStatus()
        gan_train_status.load()
        gan_generate_status.load()

        gan_train_status["model_path"] = path_to_model
        gan_generate_status["model_path"] = path_to_model
        gan_train_status.save()
        gan_generate_status.save()

        print(info.format(path_to_model))
        exit(0)
    else:  # elif argv_copy_length > 1:
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

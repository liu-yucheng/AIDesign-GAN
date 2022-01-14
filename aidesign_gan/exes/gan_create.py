""""gan create" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
# GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import pathlib
import sys

from os import path as ospath

from aidesign_gan.libs import configs
from aidesign_gan.libs import structs
from aidesign_gan.libs import utils

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exists = ospath.exists
_isdir = ospath.isdir
_Path = pathlib.Path
_stderr = sys.stderr

_CoordsConfig = configs.CoordsConfig
_DiscStruct = structs.DStruct
_FormatConfig = configs.FormatConfig
_GenStruct = structs.GStruct
_init_folder = utils.init_folder
_ModelersConfig = configs.ModelersConfig

# End of aliases

brief_usage = "gan create <path-to-model>"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info strings

info = fr"""

Created a model at {{}}

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

model_exists_info = fr"""

"{brief_usage}" finds that the model already exists
Please check the model at: {{}}
{usage}

"""
"""Info to display when the model to create already exists."""
model_exists_info = model_exists_info.strip()

model_is_not_dir_info = fr"""

"{brief_usage}" finds that the model exists but not as a directory
Please check the model at: {{}}
{usage}

"""
"""Info to display when the model exists but not as a directory."""
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

        model_exists = _exists(path_to_model)
        model_is_dir = _isdir(path_to_model)
        if model_exists and model_is_dir:
            path_to_model = str(_Path(path_to_model).resolve())
            print(model_exists_info.format(path_to_model), file=_stderr)
            exit(1)
        if model_exists and (not model_is_dir):
            path_to_model = str(_Path(path_to_model).resolve())
            print(model_is_not_dir_info.format(path_to_model), file=_stderr)
            exit(1)

        _init_folder(path_to_model)
        path_to_model = str(_Path(path_to_model).resolve())

        format_config = _FormatConfig(path_to_model)
        coords_config = _CoordsConfig(path_to_model)
        modelers_config = _ModelersConfig(path_to_model)
        d_struct = _DiscStruct(path_to_model)
        g_struct = _GenStruct(path_to_model)

        format_config.load()
        coords_config.load()
        modelers_config.load()
        d_struct.load()
        g_struct.load()

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

# Top level code


if __name__ == "__main__":
    main()

# End of top level code

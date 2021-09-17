"""Executable module for the "gan create" command."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import os
import pathlib
import sys

from aidesign_gan.libs import configs
from aidesign_gan.libs import structs
from aidesign_gan.libs import utils

# Private attributes ...

_brief_usage = "gan create <path-to-model>"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

# ... Private attributes
# Nominal info strings ...

info = r"""Created a model at {}
"""
"""The primary info to display."""

# ... Nominal info strings
# Error info strings ...

too_few_args_info = f"\"{_brief_usage}\""r""" gets too few arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
"""The info to display when the executable gets too few arguments."""

too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
"""The info to display when the executable gets too many arguments."""

model_exists_info = f"\"{_brief_usage}\""r""" finds that the model already exists
Please check the model at: {}"""fr"""
{_usage}
"""
"""The info to display when the model to create already exists."""

model_is_not_dir_info = f"\"{_brief_usage}\""r""" finds that the model exists but not as a directory
Please check the model at: {}"""fr"""
{_usage}
"""
"""The info to display when the model exists but not as a directory."""

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
    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), end="")
        exit(1)
    elif argv_copy_length == 1:
        path_to_model = "./" + argv_copy.pop(0)

        model_exists = os.path.exists(path_to_model)
        model_is_dir = os.path.isdir(path_to_model)
        if model_exists and model_is_dir:
            path_to_model = str(pathlib.Path(path_to_model).resolve())
            print(model_exists_info.format(path_to_model), end="")
            exit(1)
        if model_exists and (not model_is_dir):
            path_to_model = str(pathlib.Path(path_to_model).resolve())
            print(model_is_not_dir_info.format(path_to_model), end="")
            exit(1)

        utils.init_folder(path_to_model)
        path_to_model = str(pathlib.Path(path_to_model).resolve())

        coords_config = configs.CoordsConfig(path_to_model)
        modelers_config = configs.ModelersConfig(path_to_model)
        d_struct = structs.DStruct(path_to_model)
        g_struct = structs.GStruct(path_to_model)

        coords_config.load()
        modelers_config.load()
        d_struct.load()
        g_struct.load()

        print(info.format(path_to_model), end="")
        exit(0)
    # elif argv_copy_length > 1
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


# Let main be the script entry point
if __name__ == "__main__":
    main()

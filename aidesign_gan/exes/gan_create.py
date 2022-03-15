""""gan create" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import pathlib
import shutil
import sys

from os import path as ospath

from aidesign_gan.libs import configs
from aidesign_gan.libs import defaults

# Aliases

_argv = sys.argv
_copytree = shutil.copytree
_deepcopy = copy.deepcopy
_exists = ospath.exists
_FormatConfig = configs.FormatConfig
_join = ospath.join
_isabs = ospath.isabs
_isdir = ospath.isdir
_Path = pathlib.Path
_stderr = sys.stderr

# End of aliases

brief_usage = "gan create <path-to-model>"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

Created a model at {{}}

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

model_exists_info = fr"""

"{brief_usage}" finds that the model already exists
Please check the model at: {{}}
{usage}

""".strip()
"""Info to display when the model to create already exists."""

model_is_not_dir_info = fr"""

"{brief_usage}" finds that the model exists but not as a directory
Please check the model at: {{}}
{usage}

""".strip()
"""Info to display when the model exists but not as a directory."""

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
        path_to_model = str(argv_copy.pop(0))

        if not _isabs(path_to_model):
            path_to_model = _join(".", path_to_model)

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

        _copytree(defaults.default_gan_model_path, path_to_model, dirs_exist_ok=True)
        path_to_model = str(_Path(path_to_model).resolve())

        format_config = _FormatConfig.load_default()
        format_config = _FormatConfig.verify(format_config)
        _FormatConfig.save_to_path(format_config, path_to_model)

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

""""gan welcome" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import pkg_resources
import sys

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exit = sys.exit
_require = pkg_resources.require
_stderr = sys.stderr

# End of aliases

# Initialize _version
_version = "<unknown version>"

try:
    _packages = _require("aidesign-gan")

    if len(_packages) > 0:
        _version = _packages[0].version
except Exception as _:
    pass
# end try

brief_usage = "gan welcome"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

(-; Welcome to AIDesign-GAN {_version}! ;-)
Made by Yucheng Liu with the hope for peace in SZ.
AIDesign-GAN, AIDesign GAN modeling application.
Copyright (C) 2022 Yucheng Liu. GNU GPL3 license.

""".strip()
"""Primary info to display."""

# End of nominal info strings
# Error info strings

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 0 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too many arguments."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        print(info)
        _exit(0)
    else:  # elif argv_copy_length > 0:
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

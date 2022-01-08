""""gan welcome" command executable.

Child command of "gan."
Can be launched directly.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import pkg_resources
import sys

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_require = pkg_resources.require
_stderr = sys.stderr

# End of aliases

# Init _version
_version = "<unknown version>"
_packages = _require("aidesign-gan")
if len(_packages) > 0:
    _version = _packages[0].version

brief_usage = "gan welcome"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info strings

info = fr"""

(-; Welcome to AIDesign-GAN {_version}! ;-)

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
        print(info)
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

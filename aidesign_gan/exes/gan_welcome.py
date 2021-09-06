"""Executable for the following app parts: the "gan welcome" command.

Attributes:
    info: the primary info to display
    too_many_args_info: the info to display when the executable gets too many arguments

    argv_copy: a copy of sys.argv
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import pkg_resources
import sys

# Init _version
_version = "<unknown version>"
_packages = pkg_resources.require("aidesign-gan")
if len(_packages) > 0:
    _version = _packages[0].version

_brief_usage = "gan welcome"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info = fr"""Welcome to AIDesign-GAN {_version}! :-)
"""
too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 0 arguments; Gets {} arguments"""fr"""
{_usage}
"""

argv_copy = None


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length == 0:
        print(info, end="")
        exit(0)
    # elif argv_copy_length > 0
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

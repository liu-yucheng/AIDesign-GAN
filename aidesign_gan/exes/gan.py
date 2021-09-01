"""Executable for the following app parts: the "gan" command.

Attributes:
    info: the primary info to display
    unknown_command_info: the info to display when the executable gets an unknown command
    unknown_arg_info: the info to display when the executable gets an unknown argument
    argv_copy: a deepcopy of sys.argv, can be consumed by the commands
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import pkg_resources
import sys

from aidesign_gan.exes import gan_create
from aidesign_gan.exes import gan_dataset
from aidesign_gan.exes import gan_generate
from aidesign_gan.exes import gan_help
from aidesign_gan.exes import gan_model
from aidesign_gan.exes import gan_status
from aidesign_gan.exes import gan_train
from aidesign_gan.exes import gan_welcome


# Init _version
_version = "<unknown version>"
_packages = pkg_resources.require("aidesign-gan")
if len(_packages) > 0:
    _version = _packages[0].version

_brief_usage = "gan <command> ..."
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info: str = fr"""AIDesign-GAN (aidesign-gan) {_version}
{_usage}
"""
unknown_command_info: str = f"\"{_brief_usage}\""r""" gets an unknown command: {}"""fr"""
{_usage}
"""
unknown_arg_info: str = f"\"{_brief_usage}\""r""" gets an unknown argument: {}"""fr"""
{_usage}
"""

argv_copy: list[str] = None


def _run_command() -> None:
    global argv_copy
    assert len(argv_copy) > 0
    command = argv_copy.pop(0)
    if len(command) <= 0:
        print(unknown_command_info.format(command), end="")
        exit(1)
    elif command[0] == "-":
        print(unknown_arg_info.format(command), end="")
        exit(1)
    elif command == "help":
        gan_help.argv_copy = argv_copy
        gan_help.run()
    elif command == "welcome":
        gan_welcome.argv_copy = argv_copy
        gan_welcome.run()
    elif command == "create":
        gan_create.argv_copy = argv_copy
        gan_create.run()
    elif command == "status":
        gan_status.argv_copy = argv_copy
        gan_status.run()
    elif command == "model":
        gan_model.argv_copy = argv_copy
        gan_model.run()
    elif command == "dataset":
        gan_dataset.argv_copy = argv_copy
        gan_dataset.run()
    elif command == "generate":
        gan_generate.argv_copy = argv_copy
        gan_generate.run()
    elif command == "train":
        gan_train.argv_copy = argv_copy
        gan_train.run()
    else:
        print(unknown_command_info.format(command), end="")


def main() -> None:
    """Starts the executable."""
    global argv_copy
    argv_length = len(sys.argv)
    assert argv_length >= 1
    if argv_length == 1:
        print(info, end="")
        exit(0)
    # elif argv_length > 1
    else:
        argv_copy = copy.deepcopy(sys.argv)
        argv_copy.pop(0)
        _run_command()


# Let main be the script entry point
if __name__ == '__main__':
    main()

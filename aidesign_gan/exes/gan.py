""""gan" command executable.

AIDesign-GAN primary command.
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

brief_usage = "gan <command> ..."
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

# Nominal info strings

info = fr"""

AIDesign-GAN (aidesign-gan) {_version}
{usage}

"""
"""Primary info to display."""
info = info.strip()

# End of nominal info strings
# Error info strings

unknown_command_info = fr"""

"{brief_usage}" gets an unknown command: {{}}
{usage}

"""
"""Info to display when getting an unknown command."""
unknown_command_info = unknown_command_info.strip()

unknown_arg_info = fr"""

"{brief_usage}" gets an unknown argument: {{}}
{usage}

"""
"""Info to display when getting an unknown argument."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def _run_command():
    global argv_copy
    assert len(argv_copy) > 0
    command = argv_copy.pop(0)

    if len(command) <= 0:
        print(unknown_command_info.format(command), file=_stderr)
        exit(1)
    elif command[0] == "-":
        print(unknown_arg_info.format(command), file=_stderr)
        exit(1)
    elif command == "help":
        from aidesign_gan.exes import gan_help
        gan_help.argv_copy = argv_copy
        gan_help.run()
    elif command == "welcome":
        from aidesign_gan.exes import gan_welcome
        gan_welcome.argv_copy = argv_copy
        gan_welcome.run()
    elif command == "create":
        from aidesign_gan.exes import gan_create
        gan_create.argv_copy = argv_copy
        gan_create.run()
    elif command == "status":
        from aidesign_gan.exes import gan_status
        gan_status.argv_copy = argv_copy
        gan_status.run()
    elif command == "model":
        from aidesign_gan.exes import gan_model
        gan_model.argv_copy = argv_copy
        gan_model.run()
    elif command == "dataset":
        from aidesign_gan.exes import gan_dataset
        gan_dataset.argv_copy = argv_copy
        gan_dataset.run()
    elif command == "generate":
        from aidesign_gan.exes import gan_generate
        gan_generate.argv_copy = argv_copy
        gan_generate.run()
    elif command == "train":
        from aidesign_gan.exes import gan_train
        gan_train.argv_copy = argv_copy
        gan_train.run()
    elif command == "reset":
        from aidesign_gan.exes import gan_reset
        gan_reset.argv_copy = argv_copy
        gan_reset.run()
    else:
        print(unknown_command_info.format(command), file=_stderr)
        exit(1)
    # end if


def main():
    """Starts the executable."""
    global argv_copy
    argv_length = len(_argv)

    assert argv_length >= 1

    if argv_length == 1:
        print(info)
        exit(0)
    else:  # elif argv_length > 1:
        argv_copy = _deepcopy(_argv)
        argv_copy.pop(0)
        _run_command()

# Top level code


if __name__ == '__main__':
    main()

# End of top level code

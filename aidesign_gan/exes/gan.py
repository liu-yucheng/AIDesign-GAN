""""gan" command executable.

AIDesign-GAN primary command.
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

brief_usage = "gan <command> ..."
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

AIDesign-GAN (aidesign-gan) {_version}
{usage}

""".strip()
"""Primary info to display."""

# End of nominal info strings
# Error info strings

unknown_cmd_info = fr"""

"{brief_usage}" gets an unknown command: {{}}
{usage}

""".strip()
"""Info to display when getting an unknown command."""

unknown_arg_info = fr"""

"{brief_usage}" gets an unknown argument: {{}}
{usage}

""".strip()
"""Info to display when getting an unknown argument."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


def _run_command():
    global argv_copy

    argv_copy = [str(elem) for elem in argv_copy]

    assert len(argv_copy) > 0

    command = argv_copy.pop(0)
    command = str(command)

    if len(command) <= 0:
        print(unknown_cmd_info.format(command), file=_stderr)
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
    elif command == "export":
        from aidesign_gan.exes import gan_export
        gan_export.argv_copy = argv_copy
        gan_export.run()
    else:
        print(unknown_cmd_info.format(command), file=_stderr)
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
    # end if


if __name__ == '__main__':
    main()

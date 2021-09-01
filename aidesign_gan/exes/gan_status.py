"""Executable for the following app parts: the "gan status" command.

Attributes:
    info: the primary info to display
    too_many_args_info: the info to display when the executable gets too many arguments
    argv_copy: a copy of sys.argv
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import os
import sys

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

_brief_usage = "gan status"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info: str = r"""App data: {}
"gan train":
{}
"gan generate":
{}
"""
too_many_args_info: str = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 0 arguments; Gets {} arguments"""fr"""
{_usage}
"""

argv_copy: list[str] = None


def run() -> None:
    """Runs the executable as a command"""
    global argv_copy
    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length == 0:
        if not os.path.exists(defaults.app_data_path):
            utils.init_folder(defaults.app_data_path)

        app_data_info = defaults.app_data_path
        gan_train_info = ""
        gan_generate_info = ""

        gan_train_status = statuses.GANTrainStatus()
        gan_generate_status = statuses.GANGenerateStatus()
        gan_train_status.load()
        gan_generate_status.load()

        gan_train_lines = []
        gan_generate_lines = []
        for key in gan_train_status.items:
            tab_spaces = " " * (8 - len(key) % 8)
            gan_train_lines.append(f"    {key}:{tab_spaces}{gan_train_status[key]}")
        for key in gan_generate_status.items:
            tab_spaces = " " * (8 - len(key) % 8)
            gan_generate_lines.append(f"    {key}:{tab_spaces}{gan_generate_status[key]}")
        gan_train_info = "\n".join(gan_train_lines)
        gan_generate_info = "\n".join(gan_generate_lines)

        print(info.format(app_data_info, gan_train_info, gan_generate_info), end="")
        exit(0)
    # elif argv_copy_length > 0
    else:
        print(too_many_args_info.format(argv_copy_length), end="")
        exit(1)


def main() -> None:
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

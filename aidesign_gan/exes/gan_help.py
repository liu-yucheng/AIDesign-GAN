""""gan help" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import pydoc
import sys

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_exit = sys.exit
_pager = pydoc.pager
_stderr = sys.stderr

# End of aliases

brief_usage = "gan help"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

Usage: gan <command> ...
==== Commands ====
help:
    When:   You need help info. For example, now.
    How-to: gan help
create:
    When:   You create a new model with the defaults.
    How-to: gan create <path-to-model>
status:
    When:   You check the status of the train and generate commands.
    How-to: gan status
model:
    When:   You select the model for the next training or generation session.
    How-to: gan model <path-to-model>
dataset:
    When:   You select the dataset for the next training session.
    How-to: gan dataset <path-to-dataset>
train:
    When:   You start a training session.
    How-to: gan train
    Notes:
        You will be prompted with the command status. You need to confirm to continue. Depending on your training
        configs, the training session might take minutes, hours, or several days.
generate:
    When:   You start a generation session.
    How-to: gan generate
    Notes:
        You will be prompted with the command status. You need to confirm to continue. Depending on your generation
        configs, the generation session might take seconds or minutes.
export:
    When:   You want to export a selected model to a path for the use in other software products.
    How-to: gan export <path-to-export>
    Notes:  You can use the model export in any software product, including the proprietary ones.
reset:
    When:   You want to reset the app data, which includes the command statuses.
    How-to: gan reset
    Notes:  You will lose the current command statuses after the reset.
welcome:
    When:   You want to display the welcome message.
    How-to: gan welcome

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

# End of error info stirngs

argv_copy = None
"""Consumable copy of sys.argv."""


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        _pager(info)
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

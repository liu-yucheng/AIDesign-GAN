""""gan help" command executable.

Child command of "gan."
Can be executed directly.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import sys

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_stderr = sys.stderr

# End of aliases

brief_usage = "gan help"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

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
    Notes:  You will be prompted with the command status. You need to confirm to continue. Depending on your training
            configs, the training session might take minutes, hours, or several days.
generate:
    When:   You start a generation session.
    How-to: gan generate
    Notes:  You will be prompted with the command status. You need to confirm to continue. Depending on your generation
            configs, the generation session might take seconds or minutes.
reset:
    When:   You want to reset the app data, which contains the command statuses.
    How-to: gan reset
    Notes:  You will lose the current command statuses after the reset.
welcome:
    When:   You want to display the welcome message.
    How-to: gan welcome

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

# End of error info stirngs

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
    # elif argv_copy_length > 0:
    else:
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

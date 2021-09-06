"""Executable for the following app parts: the "gan help" command.

Attributes:
    info: the primary info to display
    too_many_args_info: the info to display when the executable gets too many arguments

    argv_copy: a copy of sys.argv
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import sys

_brief_usage = "gan help"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info = r"""Usage: gan <command> ...
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
    Notes:  You will lose the current command statuses after the reset
welcome:
    When:   You want to display the welcome message.
    How-to: gan welcome
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

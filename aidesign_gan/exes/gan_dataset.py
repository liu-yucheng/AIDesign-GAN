"""Executable for the following app parts: the "gan dataset" command.

Attributes:
    info: the primary info to display

    too_few_args_info: the info to display when the executable gets too few arguments
    too_many_args_info: the info to display when the executable gets too many arguments
    dataset_does_not_exist_info: the info to display when the selected dataset does not exist
    dataset_is_not_dir_info: the info to display when the selected dataset is not a directory

    argv_copy: a copy of sys.argv
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import os
import pathlib
import sys

from aidesign_gan.libs import statuses

_brief_usage = "gan dataset <path-to-dataset>"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info = r"""Selected the dataset at: {}
Applied the selection to "gan train"
"""

too_few_args_info = f"\"{_brief_usage}\""r""" gets too few arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
dataset_does_not_exist_info = f"\"{_brief_usage}\""r""" cannot find the dataset
Please check if the dataset is present at: {}"""fr"""
{_usage}
"""
dataset_is_not_dir_info = f"\"{_brief_usage}\""r""" finds that the dataset is not a directory
Please check if the dataset appears as a directory at: {}"""fr"""
{_usage}
"""

argv_copy = None


def run():
    """Runs the executable as a command."""
    global argv_copy
    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), end="")
        exit(1)
    elif argv_copy_length == 1:
        path_to_dataset = "./" + argv_copy.pop(0)
        path_to_dataset = str(pathlib.Path(path_to_dataset).resolve())

        if not os.path.exists(path_to_dataset):
            print(dataset_does_not_exist_info.format(path_to_dataset), end="")
            exit(1)
        if not os.path.isdir(path_to_dataset):
            print(dataset_is_not_dir_info.format(path_to_dataset), end="")
            exit(1)

        gan_train_status = statuses.GANTrainStatus()
        gan_train_status.load()

        gan_train_status["dataset_path"] = path_to_dataset
        gan_train_status.save()

        print(info.format(path_to_dataset), end="")
        exit(0)
    # elif argv_copy_length > 1
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

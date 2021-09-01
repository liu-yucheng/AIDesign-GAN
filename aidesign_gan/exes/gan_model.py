"""Executable for the following app parts: the "gan model" command.

Attributes:
    info: the primary info to display
    too_few_args_info: the info to display when the executable gets too few arguments
    too_many_args_info: the info to display when the executable gets too many arguments
    model_does_not_exist_info: the info to display when the selected model does not exist
    model_is_not_dir_info: the info to display when the selected model is not a directory
    argv_copy: a copy of sys.argv
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import os
import pathlib
import sys

from aidesign_gan.libs import statuses

_brief_usage = "gan model <path-to-model>"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info: str = r"""Selected the model at: {}
Applied the selection to "gan train" and "gan generate"
"""

too_few_args_info: str = f"\"{_brief_usage}\""r""" gets too few arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
too_many_args_info: str = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 1 arguments; Gets {} arguments"""fr"""
{_usage}
"""
model_does_not_exist_info: str = f"\"{_brief_usage}\""r""" cannot find the model
Please check if the model is present at: {}"""fr"""
{_usage}
"""
model_is_not_dir_info: str = f"\"{_brief_usage}\""r""" finds that the model is not a directory
Please check if the model appears as a directory at: {}"""fr"""
{_usage}
"""

argv_copy: list[str] = None


def run() -> None:
    """Runs the executable as a command"""
    global argv_copy
    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), end="")
        exit(1)
    elif argv_copy_length == 1:
        path_to_model = "./" + argv_copy.pop(0)
        path_to_model = str(pathlib.Path(path_to_model).resolve())

        if not os.path.exists(path_to_model):
            print(model_does_not_exist_info.format(path_to_model), end="")
            exit(1)
        if not os.path.isdir(path_to_model):
            print(model_is_not_dir_info.format(path_to_model), end="")
            exit(1)

        gan_train_status = statuses.GANTrainStatus()
        gan_generate_status = statuses.GANGenerateStatus()
        gan_train_status.load()
        gan_generate_status.load()

        gan_train_status["model_path"] = path_to_model
        gan_generate_status["model_path"] = path_to_model
        gan_train_status.save()
        gan_generate_status.save()

        print(info.format(path_to_model), end="")
        exit(0)
    # elif argv_copy_length > 1
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

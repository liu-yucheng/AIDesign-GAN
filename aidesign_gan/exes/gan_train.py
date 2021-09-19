"""Executable module for the "gan train" command."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng


import copy
import datetime
import sys
import traceback

from aidesign_gan.libs import coords
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Private attributes ...

_brief_usage = "gan train"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""
_timeout = 30

# ... Private attributes
# Nominal info strings ...

info = f"\"{_brief_usage}\":"r"""
{}
-
Please confirm the above training session setup
Do you want to continue? [ Y (Yes) | n (no) ]:"""fr""" < default: Yes, timeout: {_timeout} seconds >
"""
"""The primary info to display."""

will_start_session_info = r"""Will start a training session
---- The following will be logged to: {} ----
"""
"""The info to display when the session starts."""

completed_session_info = r"""---- The above has been logged to: {} ----
Completed the training session
"""
"""The info to display when the session completes."""

aborted_session_info = r"""Aborted the training session
"""
"""The info to display when the user aborts the session."""

# ... Nominal info strings
# Error info strings ...

too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 0 arguments; Gets {} arguments"""fr"""
{_usage}
"""
"""The info to display when the executable gets too many arguments."""

none_dataset_info = f"\"{_brief_usage}\""fr""" finds that the dataset_path selection is None
Please select a dataset with the "gan dataset <path-to-dataset>" command
{_usage}
"""
"""The info to display when the dataset selection is None."""

none_model_info = f"\"{_brief_usage}\""fr""" finds that the model_path selection is None
Please select a model with the "gan model <path-to-model>" command
{_usage}
"""
"""The info to display when the model selection is None."""

stopped_session_info = r"""---- The above has been logged to: {} ----
Stopped the training session
"""
"""The info to display when the session stops from an exception."""

# ... Error info strings
# Other public attributes ...

argv_copy = None
"""A consumable copy of sys.argv."""

dataset_path = None
"""The dataset path."""

model_path = None
"""The model path."""

log_location = None
"""The log location."""

# ... Other public attributes


def _start_session():
    global dataset_path
    global model_path
    global log_location

    start_time = datetime.datetime.now()
    log_file = open(log_location, "a+")
    all_logs = [sys.stdout, log_file]
    utils.logln(all_logs, "AIDesign-GAN training session ...")
    utils.logln(all_logs, f"Model path: {model_path}")
    utils.logln(all_logs, f"Dataset path: {dataset_path}")
    utils.logln(all_logs, "-")

    try:
        coord = coords.TrainingCoord(dataset_path, model_path, all_logs)
        coord.start_training()
    except BaseException as base_exception:
        utils.logstr(all_logs, traceback.format_exc())
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        utils.logln(all_logs, "-")
        utils.logln(all_logs, f"Execution stopped after: {execution_time} (days, hours: minutes: seconds)")
        utils.logln(all_logs, "... AIDesign-GAN training session (stopped)")
        log_file.close()
        raise base_exception

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    utils.logln(all_logs, "-")
    utils.logln(all_logs, f"Execution time: {execution_time} (days, hours: minutes: seconds)")
    utils.logln(all_logs, "... AIDesign-GAN training session")
    log_file.close()


def run():
    """Runs the executable as a command."""
    global argv_copy
    global dataset_path
    global model_path
    global log_location

    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length == 0:
        gan_train_status = statuses.GANTrainStatus()
        gan_train_status.load()

        dataset_path = gan_train_status["dataset_path"]
        model_path = gan_train_status["model_path"]

        if dataset_path is None:
            print(none_dataset_info, end="")
            exit(1)
        if model_path is None:
            print(none_model_info, end="")
            exit(1)

        gan_train_info = ""
        gan_train_lines = []
        for key in gan_train_status.items:
            tab_spaces = " " * (8 - len(key) % 8)
            gan_train_lines.append(f"    {key}:{tab_spaces}{gan_train_status[key]}")
        gan_train_info = "\n".join(gan_train_lines)

        timed_input = utils.TimedInput()
        print(info.format(gan_train_info), end="")
        answer = timed_input.take(_timeout)

        if answer is None:
            answer = "Yes"
            print(f"\n{answer} (timeout)\n", end="")
        elif len(answer) <= 0:
            answer = "Yes"
            print(f"{answer} (default)\n", end="")

        print("-\n", end="")
        if answer.lower() == "yes" or answer.lower() == "y":
            log_location = utils.find_in_path("log.txt", model_path)
            print(will_start_session_info.format(log_location), end="")

            try:
                _start_session()
            except BaseException as base_exception:
                exit_code = 1
                if isinstance(base_exception, SystemExit):
                    exit_code = base_exception.code
                print(stopped_session_info.format(log_location), end="")
                exit(exit_code)

            print(completed_session_info.format(log_location), end="")
        # elif answer.lower() == "no" or answer.lower() == "n" or any other answer
        else:
            print(aborted_session_info, end="")

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

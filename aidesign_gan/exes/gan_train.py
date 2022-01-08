""""gan train" command executable.

Child command of "gan."
Can be executed directly.
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import copy
import datetime
import sys
import traceback
import typing

from aidesign_gan.libs import coords
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_find_in_path = utils.find_in_path
_format_exc = traceback.format_exc
_IO = typing.IO
_logln = utils.logln
_logstr = utils.logstr
_now = datetime.datetime.now
_stderr = sys.stderr
_stdout = sys.stdout
_TimedInput = utils.TimedInput
_TrainCoord = coords.TrainingCoord
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan train"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

"""
"""Usage."""
usage = usage.strip()

timeout = float(30)
"""Timeout in seconds."""

# Nominal info strings

info = fr"""

"{brief_usage}":
{{}}
-
Please confirm the above training session setup
Do you want to continue? [ Y (Yes) | n (no) ]: < default: Yes, timeout: {timeout} seconds >

"""
"""Primary info to display."""
info = info.strip()

will_start_session_info = fr"""

Will start a training session
---- The following will be logged to: {{}} ----

"""
"""Info to display when the session starts."""
will_start_session_info = will_start_session_info.strip()

completed_session_info = fr"""

---- The above has been logged to: {{}} ----
Completed the training session

"""
"""Info to display when the session completes."""
completed_session_info = completed_session_info.strip()

aborted_session_info = fr"""

Aborted the training session

"""
"""Info to display when the user aborts the session."""
aborted_session_info = aborted_session_info.strip()

# End of nominal info strings
# Error info strings

too_many_args_info = fr"""
"{brief_usage}" gets too many arguments
Expects 0 arguments; Gets {{}} arguments
{usage}

"""
"""Info to display when getting too many arguments."""
too_many_args_info = too_many_args_info.strip()

none_dataset_info = fr"""

"{brief_usage}" finds that the "dataset_path" selection is None
Please select a dataset with the "gan dataset <path-to-dataset>" command
{usage}

"""
"""Info to display when the dataset selection is None."""
none_dataset_info = none_dataset_info.strip()

none_model_info = fr"""

"{brief_usage}" finds that the "model_path" selection is None
Please select a model with the "gan model <path-to-model>" command
{usage}

"""
"""Info to display when the model selection is None."""
none_model_info = none_model_info.strip()

stopped_session_info = fr"""

---- The above has been logged to: {{}} ----
Stopped the training session

"""
"""Info to display when the session stops from an exception."""

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""
dataset_path = None
"""Dataset path."""
model_path = None
"""Model path."""
log_loc = None
"""Log location."""


def _start_session():
    global dataset_path
    global model_path
    global log_loc

    start_time = _now()
    log_file: _IO = open(log_loc, "a+")
    all_logs = [_stdout, log_file]
    err_logs = [_stderr, log_file]

    start_info = fr"""

AIDesign-GAN training session
Model path:     {model_path}
Dataset path:   {dataset_path}
-

    """
    start_info = start_info.strip()

    _logln(all_logs, start_info)

    try:
        coord = _TrainCoord(dataset_path, model_path, all_logs)
        coord.start_training()
    except BaseException as base_exception:
        _logstr(err_logs, _format_exc())

        end_time = _now()
        execution_time = end_time - start_time

        stop_info = fr"""

-
Execution stopped after: {execution_time} (days, hours: minutes: seconds)
End of AIDesign-GAN training session (stopped)

        """
        stop_info = stop_info.strip()

        _logln(all_logs, stop_info)
        log_file.close()
        raise base_exception
    # end try

    end_time = _now()
    execution_time = end_time - start_time

    end_info = fr"""

-
Execution time: {execution_time} (days, hours: minutes: seconds)
End of AIDesign-GAN training session

    """
    end_info = end_info.strip()

    _logln(all_logs, end_info)
    log_file.close()


def run():
    """Runs the executable as a command."""
    global argv_copy
    global dataset_path
    global model_path
    global log_loc
    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        gan_train_status = _TrainStatus()
        gan_train_status.load()

        dataset_path = gan_train_status["dataset_path"]
        model_path = gan_train_status["model_path"]

        if dataset_path is None:
            print(none_dataset_info, file=_stderr)
            exit(1)
        if model_path is None:
            print(none_model_info, file=_stderr)
            exit(1)

        dataset_path = str(dataset_path)
        model_path = str(model_path)

        tab_width1 = 4
        tab_width2 = 8
        tab1 = " " * tab_width1
        gan_train_lines = []
        gan_train_info = ""

        for key in gan_train_status.items:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = gan_train_status[key]
            line = f"{tab1}{key}:{tab2}{val}"
            gan_train_lines.append(line)

        gan_train_info = "\n".join(gan_train_lines)

        timed_input = _TimedInput()
        print(info.format(gan_train_info))
        answer = timed_input.take(timeout)

        if answer is None:
            answer = "Yes"
            print(f"\n{answer} (timeout)\n")
        elif len(answer) <= 0:
            answer = "Yes"
            print(f"{answer} (default)\n")

        print("-\n")

        if answer.lower() == "yes" or answer.lower() == "y":
            log_loc = _find_in_path("log.txt", model_path)
            print(will_start_session_info.format(log_loc))

            try:
                _start_session()
            except BaseException as base_exception:
                exit_code = 1

                if isinstance(base_exception, SystemExit):
                    exit_code = base_exception.code

                print(stopped_session_info.format(log_loc), file=_stderr)
                exit(exit_code)
            # end try

            print(completed_session_info.format(log_loc))
        # elif answer.lower() == "no" or answer.lower() == "n" or any other answer:
        else:
            print(aborted_session_info)
        # end if

        exit(0)
    # elif argv_copy_length > 0:
    else:
        print(too_many_args_info.format(argv_copy_length), file=_stderr)
        exit(1)
    # end if


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

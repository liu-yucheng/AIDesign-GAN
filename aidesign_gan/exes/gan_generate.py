""""gan generate" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import datetime
import sys
import traceback
import typing
from os import path as ospath

from aidesign_gan.libs import coords
from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils

# Aliases

_argv = sys.argv
_deepcopy = copy.deepcopy
_join = ospath.join
_format_exc = traceback.format_exc
_GenCoord = coords.GenCoord
_GenStatus = statuses.GANGenerateStatus
_IO = typing.IO
_logln = utils.logln
_logstr = utils.logstr
_now = datetime.datetime.now
_stderr = sys.stderr
_stdout = sys.stdout
_TimedInput = utils.TimedInput

# End of aliases

brief_usage = "gan generate"
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
Please confirm the above generation session setup
Do you want to continue? [ Y (Yes) | n (no) ]: < default: Yes, timeout: {timeout} seconds >

"""
"""Primary info to display."""

info = info.strip()

will_start_session_info = fr"""

Will start a generation session
---- The following will be logged to: {{}} ----

"""
"""Info to display when the session starts."""

will_start_session_info = will_start_session_info.strip()

completed_session_info = fr"""

---- The above has been logged to: {{}} ----
Completed the generation session

"""
"""Info to display when the session completes."""

completed_session_info = completed_session_info.strip()

aborted_session_info = fr"""

Aborted the generation session

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

none_model_info = fr"""

"{brief_usage}" finds that the "model_path" selection is None
Please select a model with the "gan model <path-to-model>" command
{usage}

"""
"""Info to display when the model selection is None."""

none_model_info = none_model_info.strip()

stopped_session_info = fr"""

---- The above has been logged to: {{}} ----
Stopped the generation session

"""
"""Info to display when the session stops from an exception."""

stopped_session_info = stopped_session_info.strip()

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""
model_path = None
"""Model path."""
log_loc = None
"""Log location."""


def _start_session():
    global model_path
    global log_loc

    start_time = _now()
    log_file: _IO = open(log_loc, "a+")
    all_logs = [_stdout, log_file]
    err_logs = [_stderr, log_file]

    start_info = fr"""

AIDesign-GAN generation session
Model path: {model_path}
-

    """

    start_info = start_info.strip()
    _logln(all_logs, start_info)

    try:
        coord = _GenCoord(model_path, all_logs, debug_level=1)
        coord.prep()
        coord.start()
    except BaseException as base_exception:
        _logstr(err_logs, _format_exc())
        end_time = _now()
        execution_time = end_time - start_time

        stop_info = fr"""

-
Execution stopped after: {execution_time} (days, hours: minutes: seconds)
End of AIDesign-GAN generation session (stopped from an exception)

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
End of AIDesign-GAN generation session

    """

    end_info = end_info.strip()
    _logln(all_logs, end_info)
    log_file.close()


def run():
    """Runs the executable as a command."""
    global argv_copy
    global model_path
    global log_loc

    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        gen_status = _GenStatus.load_from_path(defaults.app_data_path)
        model_path = gen_status["model_path"]

        if model_path is None:
            print(none_model_info, file=_stderr)
            exit(1)

        model_path = str(model_path)

        tab_width1 = 4
        tab_width2 = 8
        tab1 = " " * tab_width1
        gen_lines = []
        gen_info = ""

        for key in gen_status:
            tab2 = " " * (tab_width2 - len(key) % tab_width2)
            val = gen_status[key]
            val = str(val)
            line = f"{tab1}{key}:{tab2}{val}"
            gen_lines.append(line)
        # end for

        gen_info = "\n".join(gen_lines)

        timed_input = _TimedInput()
        print(info.format(gen_info))
        answer = timed_input.take(timeout)

        if answer is None:
            answer = "Yes"
            print(f"\n{answer} (timeout)")
        elif len(answer) <= 0:
            answer = "Yes"
            print(f"{answer} (default)")
        # end if

        print("-")

        if answer.lower() == "yes" or answer.lower() == "y":
            log_loc = _join(model_path, "log.txt")
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
        else:  # elif answer.lower() == "no" or answer.lower() == "n" or any other answer:
            print(aborted_session_info)
        # end if

        exit(0)
    else:  # elif argv_copy_length > 0:
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


if __name__ == "__main__":
    main()

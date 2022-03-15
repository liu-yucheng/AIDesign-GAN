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

""".strip()
"""Usage."""

timeout = float(30)
"""Timeout in seconds."""

# Nominal info strings

info = fr"""

"{brief_usage}":
{{}}
-
Please confirm the above generation session setup
Do you want to continue? [ Y (Yes) | n (no) ]: < default: Yes, timeout: {timeout} seconds >

""".strip()
"""Primary info to display."""

will_start_session_info = fr"""

Will start a generation session
---- The following will be logged to: {{}} ----

""".strip()
"""Info to display when the session starts."""

completed_session_info = fr"""

---- The above has been logged to: {{}} ----
Completed the generation session

""".strip()
"""Info to display when the session completes."""

aborted_session_info = fr"""

Aborted the generation session

""".strip()
"""Info to display when the user aborts the session."""

# End of nominal info strings
# Error info strings

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 0 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too many arguments."""

none_model_info = fr"""

"{brief_usage}" finds that the "model_path" selection is None
Please select a model with the "gan model ..." command
{usage}

""".strip()
"""Info to display when the model selection is None."""

stopped_session_info = fr"""

---- The above has been logged to: {{}} ----
Stopped the generation session

""".strip()
"""Info to display when the session stops from an exception."""

# End of error info strings
# Session info strings

session_header_info = fr"""

AIDesign-GAN generation session
Model path: {{}}
-

""".strip()
"""Session header info."""

session_stop_trailer_info = fr"""

-
Execution stopped after: {{}} (days, hours: minutes: seconds)
End of AIDesign-GAN generation session (stopped from an exception)

""".strip()
"""Session trailer info to display after execution stops."""

session_comp_trailer_info = fr"""

-
Execution time: {{}} (days, hours: minutes: seconds)
End of AIDesign-GAN generation session

""".strip()
"""Session trailer info to display after execution completes."""

# End of session info strings

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
    _logln(all_logs, session_header_info.format(model_path))

    try:
        debug_level = 1  # NOTE: Check before each release
        coord = _GenCoord(model_path, all_logs, debug_level)
        coord.prep()
        coord.start()
    except BaseException as base_exception:
        _logstr(err_logs, _format_exc())
        end_time = _now()
        exe_time = end_time - start_time
        _logln(all_logs, session_stop_trailer_info.format(exe_time))
        log_file.close()
        raise base_exception
    # end try

    end_time = _now()
    exe_time = end_time - start_time
    _logln(all_logs, session_comp_trailer_info.format(exe_time))
    log_file.close()


def _append_status_to_lines(status, lines, tab_width1, tab_width2):
    status: dict = status
    lines: list = lines
    tab_width1 = int(tab_width1)
    tab_width2 = int(tab_width2)

    tab1 = " " * tab_width1

    for key in status:
        key = str(key)
        key_len = len(key)

        val = status[key]
        val = str(val)

        tab_actual_width2 = tab_width2 - key_len % tab_width2
        tab2 = " " * tab_actual_width2

        line = f"{tab1}{key}:{tab2}{val}"
        lines.append(line)
    # end for


def run():
    """Runs the executable as a command."""
    global argv_copy
    global model_path
    global log_loc

    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        gen_status = _GenStatus.load_from_path(defaults.app_data_path)
        gen_status = _GenStatus.verify(gen_status)

        model_path = gen_status["model_path"]

        if model_path is None:
            print(none_model_info, file=_stderr)
            exit(1)

        model_path = str(model_path)

        tab_width1 = 4
        tab_width2 = 8
        gen_lines = []
        _append_status_to_lines(gen_status, gen_lines, tab_width1, tab_width2)
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

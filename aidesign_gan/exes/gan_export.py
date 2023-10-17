""""gan export" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import datetime
import os
import pathlib
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
_exists = ospath.exists
_isabs = ospath.isabs
_join = ospath.join
_exit = sys.exit
_format_exc = traceback.format_exc
_ExportCoord = coords.ExportCoord
_ExportStatus = statuses.GANExportStatus
_IO = typing.IO
_logln = utils.logln
_logstr = utils.logstr
_makedirs = os.makedirs
_now = datetime.datetime.now
_Path = pathlib.Path
_stderr = sys.stderr
_stdout = sys.stdout
_TimedInput = utils.TimedInput

# End of aliases

brief_usage = "gan export <path-to-export>"
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
Exporting to: {{}}
-
Please confirm the above exportation session setup
Do you want to continue? [ Y (Yes) | n (no) ]: < default: Yes, timeout: {timeout} seconds >

""".strip()
"""Primary info to display."""

will_start_session_info = fr"""

Will start an exportation session
---- The following will be logged to: {{}} ----

""".strip()
"""Info to display when the session starts."""

completed_session_info = fr"""

---- The above has been logged to: {{}} ----
Completed the exportation session
You can check the export at: {{}}

""".strip()
"""Info to display when the session completes."""

aborted_session_info = fr"""

Aborted the exportation session

""".strip()
"""Info to display when the user aborts the session."""

# End of nominal info strings
# Error info strings

too_few_args_info = fr"""

"{brief_usage}" gets too few arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too few arguments."""

too_many_args_info = fr"""

"{brief_usage}" gets too many arguments
Expects 1 arguments; Gets {{}} arguments
{usage}

""".strip()
"""Info to display when getting too many arguments."""

export_path_exists_info = fr"""

"{brief_usage}" finds that export path already exists
Please check the export at: {{}}
{usage}

""".strip()
"""Info to display when the export path already exists."""

none_model_info = fr"""

"{brief_usage}" finds that the "model_path" selection is None
Please select a model with the "gan model ..." command
{usage}

""".strip()
"""Info to display when the model selection is None."""

stopped_session_info = fr"""

---- The above has been logged to: {{}} ----
Stopped the exportation session
Please check the export at: {{}}

""".strip()
"""Info to display when the session stops from an exception."""

# End of error info strings
# Session info strings

session_header_info = fr"""

AIDesign-GAN exportation session
Model path: {{}}
Export path: {{}}
-

""".strip()
"""Session header info."""

session_stop_trailer_info = fr"""

-
Execution stopped after: {{}} (days, hours: minutes: seconds)
End of AIDesign-GAN exportation session (stopped from an exception)

""".strip()
"""Session trailer info after execution stops."""

session_comp_trailer_info = fr"""

-
Execution time: {{}} (days, hours: minutes: seconds)
End of AIDesign-GAN exportation session

""".strip()
"""Session trailer info after execution completes."""

# End of session info strings

argv_copy = None
"""Consumable copy of sys.argv."""
model_path = None
"""Model path."""
export_path = None
"""Export path."""
log_loc = None
"""Log location."""


def _start_session():
    global model_path
    global export_path
    global log_loc

    start_time = _now()
    log_file: _IO = open(log_loc, "a+")
    all_logs = [_stdout, log_file]
    err_logs = [_stderr, log_file]
    _logln(all_logs, session_header_info.format(model_path, export_path))

    try:
        debug_level = 1  # NOTE: Check before each release
        coord = _ExportCoord(model_path, export_path, all_logs, debug_level)
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
    global export_path
    global log_loc

    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length < 1:
        print(too_few_args_info.format(argv_copy_length), file=_stderr)
        _exit(1)
    elif argv_copy_length == 1:
        assert argv_copy is not None
        path_to_export = str(argv_copy.pop(0))

        if not _isabs(path_to_export):
            path_to_export = _join(".", path_to_export)

        path_to_export = str(_Path(path_to_export).resolve())

        if _exists(path_to_export):
            print(export_path_exists_info.format(path_to_export), file=_stderr)
            _exit(1)

        _makedirs(path_to_export, exist_ok=True)
        export_path = path_to_export

        export_status = _ExportStatus.load_from_path(defaults.app_data_path)
        export_status = _ExportStatus.verify(export_status)

        model_path = export_status["model_path"]

        if model_path is None:
            print(none_model_info, file=_stderr)
            _exit(1)

        model_path = str(model_path)

        tab_width1 = 4
        tab_width2 = 8
        export_lines = []
        _append_status_to_lines(export_status, export_lines, tab_width1, tab_width2)
        export_info = "\n".join(export_lines)

        timed_input = _TimedInput()
        print(info.format(export_info, export_path))
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
                if isinstance(base_exception, SystemExit):
                    exit_code = base_exception.code
                else:
                    exit_code = 1

                print(stopped_session_info.format(log_loc, export_path), file=_stderr)
                _exit(exit_code)
            # end try

            print(completed_session_info.format(log_loc, export_path))
        else:  # elif answer.lower() == "no" or answer.lower() == "n" or any other answer:
            print(aborted_session_info)
        # end if

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

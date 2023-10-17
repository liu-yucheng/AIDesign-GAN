""""gan status" command executable.

Child command of "gan."
Can be launched directly.
"""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import copy
import shutil
import sys

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import statuses

# Aliases

_argv = sys.argv
_copytree = shutil.copytree
_deepcopy = copy.deepcopy
_exists = ospath.exists
_exit = sys.exit
_ExportStatus = statuses.GANExportStatus
_GenStatus = statuses.GANGenerateStatus
_stderr = sys.stderr
_TrainStatus = statuses.GANTrainStatus

# End of aliases

brief_usage = "gan status"
"""Brief usage."""

usage = fr"""

Usage: {brief_usage}
Help: gan help

""".strip()
"""Usage."""

# Nominal info strings

info = fr"""

App data is at: {{}}
"gan train" status:
{{}}
"gan generate" status:
{{}}
"gan export ..." status:
{{}}

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

# End of error info strings

argv_copy = None
"""Consumable copy of sys.argv."""


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
    argv_copy_length = len(argv_copy)

    assert argv_copy_length >= 0

    if argv_copy_length == 0:
        if not _exists(defaults.app_data_path):
            _copytree(defaults.default_app_data_path, defaults.app_data_path, dirs_exist_ok=True)

        train_status = _TrainStatus.load_from_path(defaults.app_data_path)
        gen_status = _GenStatus.load_from_path(defaults.app_data_path)
        export_status = _ExportStatus.load_from_path(defaults.app_data_path)

        train_lines = []
        gen_lines = []
        export_lines = []

        tab_width1 = 4
        tab_width2 = 8

        _append_status_to_lines(train_status, train_lines, tab_width1, tab_width2)
        _append_status_to_lines(gen_status, gen_lines, tab_width1, tab_width2)
        _append_status_to_lines(export_status, export_lines, tab_width1, tab_width2)

        app_data_info = defaults.app_data_path
        train_info = "\n".join(train_lines)
        gen_info = "\n".join(gen_lines)
        export_info = "\n".join(export_lines)

        print(info.format(app_data_info, train_info, gen_info, export_info))
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

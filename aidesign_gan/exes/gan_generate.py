"""Executable for the following app parts: the "gan generate" command.

Attributes:
    info: the primary info to display
    will_start_session_info: the info to display when the session starts
    completed_session_info: the info to display when the session completes
    aborted_session_info: the info to display when the user aborts the session

    too_many_args_info: the info to display when the executable gets too many arguments
    none_model_info: the info to display when the model selection is None

    argv_copy: a copy of sys.argv
    model_path: the model path
    log_location: the log location
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng


import copy
import datetime
import sys
import traceback

from aidesign_gan.libs import coords
from aidesign_gan.libs import statuses
from aidesign_gan.libs import utils


_brief_usage = "gan generate"
_usage = fr"""Usage: {_brief_usage}
Help: gan help"""

info = f"\"{_brief_usage}\":"r"""
{}
Please confirm the above generation session setup
Do you want to continue? [ Y (Yes) | n (no) ]:    """
will_start_session_info = r"""Will start a generation session
---- The following will be logged to: {} ----
"""
completed_session_info = r"""---- The above has been logged to: {} ----
Completed the generation session
"""
aborted_session_info = r"""Aborted the generation session
"""

too_many_args_info = f"\"{_brief_usage}\""r""" gets too many arguments
Expects 0 arguments; Gets {} arguments"""fr"""
{_usage}
"""
none_model_info = f"\"{_brief_usage}\""fr""" finds that the model_path selection is None
Please select a model with the "gan model <path-to-model>" command
{_usage}
"""

argv_copy = None
model_path = None
log_location = None


def start_session():
    """Starts a generation session."""
    global model_path
    global log_location

    start_time = datetime.datetime.now()
    log_file = open(log_location, "a+")
    all_logs = [sys.stdout, log_file]
    utils.logln(all_logs, "AIDesign-GAN generation session ...")
    utils.logln(all_logs, f"Model path: {model_path}")
    utils.logln(all_logs, "-")

    try:
        coord = coords.GenerationCoord(model_path, all_logs)
        coord.setup_results()
        coord.setup_context()
        coord.start_generation()
    except Exception as _:
        utils.logstr(all_logs, traceback.format_exc())
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        utils.logln(all_logs, "-")
        utils.logln(all_logs, f"Execution stopped after: {execution_time} (days, hours: minutes: seconds)")
        utils.logln(all_logs, "... AIDesign-GAN generation session (stopped from an exception)")
        log_file.close()
        exit(1)

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    utils.logln(all_logs, "-")
    utils.logln(all_logs, f"Execution time: {execution_time} (days, hours: minutes: seconds)")
    utils.logln(all_logs, "... AIDesign-GAN generation session")
    log_file.close()


def run():
    """Runs the executable as a command."""
    global argv_copy
    global model_path
    global log_location

    argv_copy_length = len(argv_copy)
    assert argv_copy_length >= 0
    if argv_copy_length == 0:
        gan_generate_status = statuses.GANGenerateStatus()
        gan_generate_status.load()

        model_path = gan_generate_status["model_path"]
        if model_path is None:
            print(none_model_info, end="")
            exit(1)

        gan_generate_info = ""
        gan_generate_lines = []
        for key in gan_generate_status.items:
            tab_spaces = " " * (8 - len(key) % 8)
            gan_generate_lines.append(f"    {key}:{tab_spaces}{gan_generate_status[key]}")
        gan_generate_info = "\n".join(gan_generate_lines)

        print(info.format(gan_generate_info), end="")
        answer = input()
        if len(answer) <= 0:
            answer = "Yes"
            print(f"{answer} (default)\n", end="")

        if answer.lower() == "yes" or answer.lower() == "y":
            log_location = utils.find_in_path("log.txt", model_path)
            print(will_start_session_info.format(log_location), end="")
            start_session()
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

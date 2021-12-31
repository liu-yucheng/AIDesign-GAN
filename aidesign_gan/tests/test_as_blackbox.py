"""Executable that tests the app as a blackbox."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import asyncio
import os
import pathlib
import threading
import typing
import unittest

from os import path

_create_subprocess_shell = asyncio.create_subprocess_shell
_IO = typing.IO
_join = path.join
_makedirs = os.makedirs
_Path = pathlib.Path
_PIPE = asyncio.subprocess.PIPE
_run = asyncio.run
_TestCase = unittest.TestCase
_Thread = threading.Thread

# Private attributes.

_timeout = float(30)

_aidesign_gan_tests_path = str(_Path(__file__).parent)
_aidesign_gan_repo_path = str(_Path(_aidesign_gan_tests_path).parent.parent)
_aidesign_gan_test_data_path = _join(_aidesign_gan_repo_path, ".aidesign_gan_test_data")
_logloc = _join(_aidesign_gan_test_data_path, "log.txt")


def _fix_newline_format(instr):
    instr = str(instr)

    result = instr
    result = result.replace("\r\n", "\n")
    result = result.replace("\r", "\n")
    return result


async def _async_run_cmd(cmd, instr=""):
    cmd = str(cmd)
    instr = str(instr)

    subproc = await _create_subprocess_shell(cmd=cmd, stdin=_PIPE, stdout=_PIPE, stderr=_PIPE)
    inbytes = instr.encode("utf-8", "replace")
    out, err = await subproc.communicate(inbytes)
    exit_code = await subproc.wait()

    out = out.decode("utf-8", "replace")
    out = _fix_newline_format(out)
    err = err.decode("utf-8", "replace")
    err = _fix_newline_format(err)
    result = exit_code, out, err
    return result


def _run_cmd(cmd, instr=""):
    cmd = str(cmd)
    instr = str(instr)

    result = _run(_async_run_cmd(cmd, instr))
    return result


class _FuncThread(_Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        """Runs the thread."""
        # Adopted from CPython standard library threading source code
        # Ref: https://github.com/python/cpython/blob/main/Lib/threading.py
        try:
            if self._target is not None:
                self._result = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid reference cycles
            del self._target
            del self._args
            del self._kwargs

    def join(self, timeout=None):
        """Joins the thread.

        Returns:
            self._result: the result
        """
        super().join(timeout=timeout)
        return self._result


class _TestCmd(_TestCase):

    def __init__(self, methodName=""):
        super().__init__(methodName=methodName)
        self._log: _IO = None

    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        _makedirs(_aidesign_gan_test_data_path, exist_ok=True)
        self._log = open(_logloc, "a+")

        start_info = str(
            "\n"
            "- Test case {}\n"
            "\n"
        ).format(type(self).__name__)
        self._log.write(start_info)

    def tearDown(self):
        """Tears down the test case."""
        super().tearDown()
        end_info = str(
            "\n"
            "- End of test case {}\n"
            "\n"
        ).format(type(self).__name__)
        self._log.write(end_info)

        self._log.flush()
        self._log.close()


# End of private attributes.
# Public attributes.


class TestGAN(_TestCmd):
    """Tests for the "gan" command."""

    def __init__(self, methodName=""):
        super().__init__(methodName=methodName)
        self._log = None

    def test_norm(self):
        """Tests "gan" normal use case."""
        start_info = "-- Test method {}\n".format(self.test_norm.__name__)
        self._log.write(start_info)

        cmd = "gan"
        thread = _FuncThread(target=_run_cmd, args=[cmd, ""])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        out_info = str(
            "--- \"{}\" stdout\n"
            "{}\n"
            "--- End of \"{}\" stdout\n"
        ).format(
            cmd,
            out,
            cmd
        )
        self._log.write(out_info)

        err_info = str(
            "--- \"{}\" stderr\n"
            "{}\n"
            "--- End of \"{}\" stderr\n"
        ).format(
            cmd,
            err,
            cmd
        )
        self._log.write(err_info)

        fail_msg = "Running \"{}\" results in an timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        end_info = "-- End of test method {}\n".format(self.test_norm.__name__)
        self._log.write(end_info)


class TestGANHelp(_TestCmd):
    """Tests for the "gan help" command."""

    def __init__(self, methodName=""):
        super().__init__(methodName=methodName)
        self._log = None

    def test_norm(self):
        """Tests "gan help" normal use case."""
        start_info = "-- Test method {}\n".format(self.test_norm.__name__)
        self._log.write(start_info)

        cmd = "gan help"
        thread = _FuncThread(target=_run_cmd, args=[cmd, ""])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        out_info = str(
            "--- \"{}\" stdout\n"
            "{}\n"
            "--- End of \"{}\" stdout\n"
        ).format(
            cmd,
            out,
            cmd
        )
        self._log.write(out_info)

        err_info = str(
            "--- \"{}\" stderr\n"
            "{}\n"
            "--- End of \"{}\" stderr\n"
        ).format(
            cmd,
            err,
            cmd
        )
        self._log.write(err_info)

        fail_msg = "Running \"{}\" results in an timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        end_info = "-- End of test method {}\n".format(self.test_norm.__name__)
        self._log.write(end_info)


def main():
    """Runs this module as an executable."""
    unittest.main(verbosity=1)


if __name__ == "__main__":
    main()

# End of public attributes.

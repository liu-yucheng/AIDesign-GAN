"""Executable that tests the app as a blackbox."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import asyncio
import os
import pathlib
import shutil
import threading
import typing
import unittest

from os import path

_create_subprocess_shell = asyncio.create_subprocess_shell
_IO = typing.IO
_isdir = path.isdir
_isfile = path.isfile
_join = path.join
_listdir = os.listdir
_makedirs = os.makedirs
_Path = pathlib.Path
_PIPE = asyncio.subprocess.PIPE
_rmtree = shutil.rmtree
_run = asyncio.run
_TestCase = unittest.TestCase
_Thread = threading.Thread

# Private attributes.

_timeout = float(60)

_aidesign_gan_tests_path = str(_Path(__file__).parent)
_aidesign_gan_repo_path = str(_Path(_aidesign_gan_tests_path).parent.parent)
_aidesign_gan_test_data_path = _join(_aidesign_gan_repo_path, ".aidesign_gan_test_data")

_log_loc = _join(_aidesign_gan_test_data_path, "log.txt")
_model_path = _join(_aidesign_gan_test_data_path, "test_model")
_model_fnames = [
    "coords_config.json",
    "discriminator_struct.py",
    "format_config.json",
    "generator_struct.py",
    "modelers_config.json"
]


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
        self._log = open(_log_loc, "a+")

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

    def _logstr(self, str_to_log):
        str_to_log = str(str_to_log)

        if self._log is not None:
            self._log.write(str_to_log)

    def _log_method_start(self, method_name):
        method_name = str(method_name)

        info = "-- Test method {}\n".format(method_name)
        self._logstr(info)

    def _log_method_end(self, method_name):
        method_name = str(method_name)

        info = "-- End of test method {}\n".format(method_name)
        self._logstr(info)

    def _log_cmdout_start(self, cmd, stream_name):
        cmd = str(cmd)
        stream_name = str(stream_name)

        info = "--- \"{}\" {}\n".format(cmd, stream_name)
        self._logstr(info)

    def _log_cmdout_end(self, cmd, stream_name):
        cmd = str(cmd)
        stream_name = str(stream_name)

        info = "--- End of \"{}\" {}\n".format(cmd, stream_name)
        self._logstr(info)


class _TestSimpleCmd(_TestCmd):

    def _test_cmd_norm(self, cmd, instr=""):
        cmd = str(cmd)
        instr = str(instr)

        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout_start(cmd, "stdout")
        outinfo = "{}\n".format(out)
        self._logstr(outinfo)
        self._log_cmdout_end(cmd, "stdout")

        self._log_cmdout_start(cmd, "stderr")
        errinfo = "{}\n".format(err)
        self._logstr(errinfo)
        self._log_cmdout_end(cmd, "stderr")

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

# End of private attributes.
# Public attributes.


class TestGAN(_TestSimpleCmd):
    """Tests for the "gan" command."""

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        cmd = "gan"
        instr = ""
        self._log_method_start(method_name)
        self._test_cmd_norm(cmd, instr)
        self._log_method_end(method_name)


class TestGANHelp(_TestSimpleCmd):
    """Tests for the "gan help" command."""

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        cmd = "gan help"
        instr = ""
        self._log_method_start(method_name)
        self._test_cmd_norm(cmd, instr)
        self._log_method_end(method_name)


class TestGANStatus(_TestSimpleCmd):
    """Tests for the "gan status" command."""

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        cmd = "gan status"
        instr = ""
        self._log_method_start(method_name)
        self._test_cmd_norm(cmd, instr)
        self._log_method_end(method_name)


class TestGANReset(_TestSimpleCmd):
    """Tests for the "gan reset" command."""

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        cmd = "gan reset"
        instr = ""
        self._log_method_start(method_name)
        self._test_cmd_norm(cmd, instr)
        self._log_method_end(method_name)


class TestGANWelcome(_TestSimpleCmd):
    """Tests for the "gan welcome" command."""

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        cmd = "gan welcome"
        instr = ""
        self._log_method_start(method_name)
        self._test_cmd_norm(cmd, instr)
        self._log_method_end(method_name)


class TestGANCreate(_TestCmd):
    """Tests for the "gan create" command."""

    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        _rmtree(_model_path, ignore_errors=True)

    def test_norm(self):
        method_name = self.test_norm.__name__
        cmd = "gan create {}".format(_model_path)
        instr = ""
        self._log_method_start(method_name)

        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout_start(cmd, "stdout")
        outinfo = "{}\n".format(out)
        self._logstr(outinfo)
        self._log_cmdout_end(cmd, "stdout")

        self._log_cmdout_start(cmd, "stderr")
        errinfo = "{}\n".format(err)
        self._logstr(errinfo)
        self._log_cmdout_end(cmd, "stderr")

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        format_incorrect_info = "model format incorrect"

        isdir = _isdir(_model_path)
        fail_msg = "{} is not a directory; {}".format(_model_path, format_incorrect_info)
        self.assertTrue(isdir, fail_msg)

        contents = _listdir(_model_path)
        for fname in _model_fnames:
            exists = fname in contents
            fail_msg = "{} is not in {}; {}".format(fname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            loc = _join(_model_path, fname)
            isfile = _isfile(loc)
            fail_msg = "{} is not a file; {}".format(loc, format_incorrect_info)
            self.assertTrue(isfile, fail_msg)

        self._log_method_end(method_name)

    def tearDown(self):
        """Tears down the test case."""
        super().tearDown()
        _rmtree(_model_path, ignore_errors=True)


def main():
    """Runs this module as an executable."""
    unittest.main(verbosity=1)


if __name__ == "__main__":
    main()

# End of public attributes.

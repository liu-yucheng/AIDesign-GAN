"""Executable that tests the app as a blackbox."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import asyncio
import json
import os
import pathlib
import shutil
import threading
import typing
import unittest

from os import path as ospath

# - Private attributes
# -- Aliases

_copytree = shutil.copytree
_create_subprocess_shell = asyncio.create_subprocess_shell
_dump = json.dump
_exists = ospath.exists
_IO = typing.IO
_isdir = ospath.isdir
_isfile = ospath.isfile
_join = ospath.join
_listdir = os.listdir
_load = json.load
_makedirs = os.makedirs
_Path = pathlib.Path
_PIPE = asyncio.subprocess.PIPE
_remove = os.remove
_rmtree = shutil.rmtree
_run = asyncio.run
_TestCase = unittest.TestCase
_Thread = threading.Thread

# -- End of aliases

_timeout = float(60)

_tests_path = str(_Path(__file__).parent)
_repo_path = str(_Path(_tests_path).parent.parent)
_test_data_path = _join(_repo_path, ".aidesign_gan_test_data")

_default_configs_path = _join(_repo_path, "aidesign_gan_default_configs")
_default_test_data_path = _join(_default_configs_path, "test_data")

_app_data_path = _join(_repo_path, ".aidesign_gan_app_data")

_log_loc = _join(_test_data_path, "log.txt")
_model_path = _join(_test_data_path, "test_model")
_dataset_path = _join(_test_data_path, "test_dataset")
_default_model_path = _join(_default_test_data_path, "test_model")
_default_dataset_path = _join(_default_test_data_path, "test_dataset")
_train_status_loc = _join(_app_data_path, "gan_train_status.json")
_generate_status_loc = _join(_app_data_path, "gan_generate_status.json")
_train_status_backup_loc = _join(_test_data_path, "gan_train_status_backup.json")
_generate_status_backup_loc = _join(_test_data_path, "gan_generate_status_backup.json")


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
        """Inits self with the given args.

        Args:
            group: Group.
            target: Target.
            name: Name.
            args: Arguments
            kwargs: Keyword arguments.
            *: Variable arguments.
            daemon: Daemon thread switch.
        """
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

        Args:
            timeout: Timeout length in seconds.

        Returns:
            self._result: Result. Return value.
        """
        super().join(timeout=timeout)
        return self._result


class _TestCmd(_TestCase):

    def __init__(self, methodName=""):
        """Inits self with the given args.

        Args:
            methodName: Method name.
        """
        super().__init__(methodName=methodName)
        self._log: _IO = None

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        _makedirs(_test_data_path, exist_ok=True)
        self._log = open(_log_loc, "a+")

        start_info = str(
            "\n"
            "- Test case {}\n"
            "\n"
        ).format(type(self).__name__)
        self._log.write(start_info)

    def tearDown(self):
        """Tears down after the tests."""
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

    def _log_cmdout(self, cmd, stream_name, out):
        cmd = str(cmd)
        stream_name = str(stream_name)
        out = str(out)

        self._log_cmdout_start(cmd, stream_name)

        out_info = "{}\n".format(out)
        self._logstr(out_info)

        self._log_cmdout_end(cmd, stream_name)

    def _backup_app_data(self):
        train_status = _load_json(_train_status_loc)
        generate_status = _load_json(_generate_status_loc)
        _save_json(train_status, _train_status_backup_loc)
        _save_json(generate_status, _generate_status_backup_loc)

    def _restore_app_data(self):
        train_status = _load_json(_train_status_backup_loc)
        generate_status = _load_json(_generate_status_backup_loc)
        _save_json(train_status, _train_status_loc)
        _save_json(generate_status, _generate_status_loc)

        if _exists(_train_status_backup_loc):
            _remove(_train_status_backup_loc)
        if _exists(_generate_status_backup_loc):
            _remove(_generate_status_backup_loc)


class _TestSimpleCmd(_TestCmd):

    def _test_cmd_norm(self, cmd, instr=""):
        cmd = str(cmd)
        instr = str(instr)

        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)


def _load_json(from_loc):
    from_loc = str(from_loc)

    file = open(from_loc, "r")
    result = _load(file)
    file.close()

    result = dict(result)
    return result


def _save_json(from_dict, to_loc):
    from_dict = dict(from_dict)
    to_loc = str(to_loc)

    file = open(to_loc, "w+")
    _dump(from_dict, file, indent=4)
    file.close()

# - End of private attributes
# - Public attributes


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

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        self._backup_app_data()

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        self._restore_app_data()

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
        """Sets up before the tests."""
        super().setUp()
        _rmtree(_model_path, ignore_errors=True)

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        _rmtree(_model_path, ignore_errors=True)

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        self._log_method_start(method_name)

        cmd = "gan create {}".format(_model_path)
        instr = ""
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        format_incorrect_info = "model format incorrect"

        isdir = _isdir(_model_path)
        fail_msg = "{} is not a directory; {}".format(_model_path, format_incorrect_info)
        self.assertTrue(isdir, fail_msg)

        model_fnames = [
            "coords_config.json",
            "discriminator_struct.py",
            "format_config.json",
            "generator_struct.py",
            "modelers_config.json"
        ]
        contents = _listdir(_model_path)
        for fname in model_fnames:
            exists = fname in contents
            fail_msg = "{} is not in {}; {}".format(fname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            loc = _join(_model_path, fname)
            isfile = _isfile(loc)
            fail_msg = "{} is not a file; {}".format(loc, format_incorrect_info)
            self.assertTrue(isfile, fail_msg)

        self._log_method_end(method_name)


class TestGANModel(_TestCmd):
    """Tests for the "gan model" command."""

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        _rmtree(_model_path, ignore_errors=True)
        _copytree(_default_model_path, _model_path, dirs_exist_ok=True)
        self._backup_app_data()

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        _rmtree(_model_path, ignore_errors=True)
        self._restore_app_data()

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        self._log_method_start(method_name)

        cmd = "gan model {}".format(_model_path)
        instr = ""
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        status = _load_json(_train_status_loc)
        key = "model_path"
        value = status[key]
        value = str(value)
        path_correct = value == _model_path
        fail_msg = "Config {} key {} has value {} but not the expected {}".format(
            _train_status_loc, key, value, _model_path
        )
        self.assertTrue(path_correct, fail_msg)

        status = _load_json(_generate_status_loc)
        key = "model_path"
        value = status[key]
        value = str(value)
        path_correct = value == _model_path
        fail_msg = "Config {} key {} has value {} but not the expected {}".format(
            _generate_status_loc, key, value, _model_path
        )
        self.assertTrue(path_correct, fail_msg)

        self._log_method_end(method_name)


class TestGANDataset(_TestCmd):
    """Tests for the "gan dataset" command."""

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        _rmtree(_dataset_path, ignore_errors=True)
        _copytree(_default_dataset_path, _dataset_path, dirs_exist_ok=True)
        self._backup_app_data()

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        _rmtree(_dataset_path, ignore_errors=True)
        self._restore_app_data()

    def test_norm(self):
        """Tests the normal use case."""
        method_name = self.test_norm.__name__
        self._log_method_start(method_name)

        cmd = "gan dataset {}".format(_dataset_path)
        instr = ""
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        status = _load_json(_train_status_loc)
        key = "dataset_path"
        value = status[key]
        value = str(value)
        path_correct = value == _dataset_path
        fail_msg = "Config {} key {} has value {} but not the expected {}".format(
            _train_status_loc, key, value, _dataset_path
        )
        self.assertTrue(path_correct, fail_msg)

        self._log_method_end(method_name)


class TestGANTrain(_TestCmd):
    """Tests for the "gan train" command."""

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        self._backup_app_data()

        _rmtree(_dataset_path, ignore_errors=True)
        _rmtree(_model_path, ignore_errors=True)
        _copytree(_default_dataset_path, _dataset_path)
        _copytree(_default_model_path, _model_path)

        _train_status = _load_json(_train_status_loc)
        _train_status["dataset_path"] = _dataset_path
        _train_status["model_path"] = _model_path
        _save_json(_train_status, _train_status_loc)

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        self._restore_app_data()

        _rmtree(_dataset_path, ignore_errors=True)
        _rmtree(_model_path, ignore_errors=True)

    def test_normal(self):
        """Tests the normal use case."""
        method_name = self.test_normal.__name__
        self._log_method_start(method_name)

        cmd = "gan train"
        instr = "\n"
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        format_incorrect_info = "training saves and results format incorrect"

        isdir = _isdir(_model_path)
        fail_msg = "{} is not a directory; {}".format(_model_path, format_incorrect_info)
        self.assertTrue(isdir, fail_msg)

        model_dnames = ["Training-Results"]
        model_fnames = [
            "coords_config.json",
            "discriminator_optim.pt",
            "discriminator_state.pt",
            "discriminator_struct.py",
            "format_config.json",
            "generator_optim.pt",
            "generator_state.pt",
            "generator_struct.py",
            "log.txt",
            "modelers_config.json"
        ]

        contents = _listdir(_model_path)

        for dname in model_dnames:
            exists = dname in contents
            fail_msg = "{} is not in {}; {}".format(dname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            path = _join(_model_path, dname)
            isdir = _isdir(path)
            fail_msg = "{} is not a directory; {}".format(path, format_incorrect_info)
            self.assertTrue(isdir, fail_msg)

        for fname in model_fnames:
            exists = fname in contents
            fail_msg = "{} is not in {}; {}".format(fname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            loc = _join(_model_path, fname)
            isfile = _isfile(loc)
            fail_msg = "{} is not a file; {}".format(loc, format_incorrect_info)
            self.assertTrue(isfile, fail_msg)

        self._log_method_end(method_name)


class TestGANGenerate(_TestCmd):
    """Tests for the "gan generate" command.

    NOTE: Relies on the correctly working "gan train" command.
    NOTE: If "gan train" works incorrectly, the testing result is undefined.
    """

    def setUp(self):
        """Sets up before the tests."""
        super().setUp()
        self._backup_app_data()

        _rmtree(_dataset_path, ignore_errors=True)
        _rmtree(_model_path, ignore_errors=True)
        _copytree(_default_dataset_path, _dataset_path)
        _copytree(_default_model_path, _model_path)

        _train_status = _load_json(_train_status_loc)
        _train_status["dataset_path"] = _dataset_path
        _train_status["model_path"] = _model_path
        _save_json(_train_status, _train_status_loc)

        _generate_status = _load_json(_generate_status_loc)
        _generate_status["model_path"] = _model_path
        _save_json(_generate_status, _generate_status_loc)

    def tearDown(self):
        """Tears down after the tests."""
        super().tearDown()
        self._restore_app_data()

        _rmtree(_dataset_path, ignore_errors=True)
        _rmtree(_model_path, ignore_errors=True)

    def test_normal(self):
        """Tests the normal use case."""
        method_name = self.test_normal.__name__
        self._log_method_start(method_name)

        # Run "gan train" first

        cmd = "gan train"
        instr = "\n"
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        # Run "gan generate" after running "gan train"

        cmd = "gan generate"
        instr = "\n"
        thread = _FuncThread(target=_run_cmd, args=[cmd, instr])
        thread.start()
        exit_code, out, err = thread.join(_timeout)
        timed_out = thread.is_alive()

        self._log_cmdout(cmd, "stdout", out)
        self._log_cmdout(cmd, "stderr", err)

        # Test "gan generate" results"

        fail_msg = "Running \"{}\" results in a timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)

        format_incorrect_info = "Generation results format incorrect"

        isdir = _isdir(_model_path)
        fail_msg = "{} is not a directory; {}".format(_model_path, format_incorrect_info)
        self.assertTrue(isdir, fail_msg)

        model_dnames = ["Generation-Results"]
        model_fnames = [
            "coords_config.json",
            "discriminator_optim.pt",
            "discriminator_state.pt",
            "discriminator_struct.py",
            "format_config.json",
            "generator_optim.pt",
            "generator_state.pt",
            "generator_struct.py",
            "log.txt",
            "modelers_config.json"
        ]

        contents = _listdir(_model_path)

        for dname in model_dnames:
            exists = dname in contents
            fail_msg = "{} is not in {}; {}".format(dname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            path = _join(_model_path, dname)
            isdir = _isdir(path)
            fail_msg = "{} is not a directory; {}".format(path, format_incorrect_info)
            self.assertTrue(isdir, fail_msg)

        for fname in model_fnames:
            exists = fname in contents
            fail_msg = "{} is not in {}; {}".format(fname, _model_path, format_incorrect_info)
            self.assertTrue(exists, fail_msg)

            loc = _join(_model_path, fname)
            isfile = _isfile(loc)
            fail_msg = "{} is not a file; {}".format(loc, format_incorrect_info)
            self.assertTrue(isfile, fail_msg)

        self._log_method_end(method_name)


def main():
    """Runs this module as an executable."""
    unittest.main(verbosity=1)

# - End of public attributes
# - Top level code


if __name__ == "__main__":
    main()

# - End of top level code

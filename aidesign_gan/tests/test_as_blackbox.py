"""Executable that tests the app as a blackbox."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import asyncio
import threading
import unittest

_create_subprocess_shell = asyncio.create_subprocess_shell
_PIPE = asyncio.subprocess.PIPE
_run = asyncio.run
_TestCase = unittest.TestCase
_Thread = threading.Thread


_timeout = float(0.001)


async def _async_run_cmd(cmd):
    cmd = str(cmd)

    subproc = await _create_subprocess_shell(cmd=cmd, stdin=_PIPE, stdout=_PIPE, stderr=_PIPE)
    result = await subproc.wait()
    return result


def _run_cmd(cmd):
    cmd = str(cmd)

    result = _run(_async_run_cmd(cmd))
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


class TestGANCmd(_TestCase):
    """Tests for the "gan" command."""

    def test_normal(self):
        """Tests normal use case."""
        cmd = "gan"
        thread = _FuncThread(target=_run_cmd, args=[cmd])
        thread.start()
        exit_code = thread.join(_timeout)
        timed_out = thread.is_alive()

        fail_msg = "Running \"{}\" results in an timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)


def main():
    """Runs this module as an executable."""
    unittest.main()


if __name__ == "__main__":
    main()

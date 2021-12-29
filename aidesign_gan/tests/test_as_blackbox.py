"""Executable that tests the app as a blackbox."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import asyncio
import unittest

_create_subprocess_shell = asyncio.create_subprocess_shell
_PIPE = asyncio.subprocess.PIPE
_run = asyncio.run
_TestCase = unittest.TestCase
_TimeoutError = asyncio.TimeoutError
_wait_for = asyncio.wait_for


_timeout = float(30)


async def _run_cmd(cmd):
    cmd = str(cmd)

    subproc = await _create_subprocess_shell(cmd=cmd, stdin=_PIPE, stdout=_PIPE, stderr=_PIPE)
    result = await subproc.wait()
    return result


async def _timed_run_cmd(cmd, timeout):
    cmd = str(cmd)
    timeout = float(timeout)

    try:
        exit_code = await _wait_for(_run_cmd(cmd), timeout=timeout)
        timed_out = False
    except _TimeoutError:
        exit_code = None
        timed_out = True

    result = exit_code, timed_out
    return result


class GAN(_TestCase):
    """Tests for the exes.gan module."""

    def test_cmd(self):
        """Tests the "gan" command."""
        cmd = "gan"
        outcome = _run(_timed_run_cmd(cmd, _timeout))
        exit_code, timed_out = outcome

        fail_msg = "Running \"{}\" results in an timeout".format(cmd)
        self.assertTrue(timed_out is False, fail_msg)

        fail_msg = "Running \"{}\" results in an unexpected exit code: {}".format(cmd, exit_code)
        self.assertTrue(exit_code == 0, fail_msg)


def main():
    """Runs this module as an executable."""
    unittest.main()


if __name__ == "__main__":
    main()

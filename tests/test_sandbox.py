"""Tests for subprocess sandbox: execution, timeout, error capture, resource limits.

All subprocess.run calls are mocked — no real code is executed.
"""

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from src.sandbox import CodeSandbox, ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_proc(stdout: str = "1 passed", stderr: str = "", returncode: int = 0):
    """Build a mock CompletedProcess."""
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


@pytest.fixture
def sandbox():
    """CodeSandbox with unshare disabled (avoids a real subprocess call in __init__)."""
    with patch("src.sandbox._check_unshare_available", return_value=False):
        return CodeSandbox()


# ---------------------------------------------------------------------------
# ExecutionResult defaults
# ---------------------------------------------------------------------------

def test_execution_result_defaults():
    r = ExecutionResult(success=True, output="ok", all_passed=True)
    assert r.timed_out is False


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

@patch("src.sandbox.subprocess.run")
def test_execute_all_passed(mock_run, sandbox):
    mock_run.return_value = make_proc("1 passed", returncode=0)
    result = sandbox.execute("x = 1", "def test_x(): assert x == 1")
    assert result.success is True
    assert result.all_passed is True
    assert "passed" in result.output


@patch("src.sandbox.subprocess.run")
def test_execute_test_failure(mock_run, sandbox):
    mock_run.return_value = make_proc("1 failed, 0 passed", returncode=1)
    result = sandbox.execute("x = 2", "def test_x(): assert x == 1")
    assert result.success is True      # subprocess succeeded; tests failed
    assert result.all_passed is False


@patch("src.sandbox.subprocess.run")
def test_execute_returns_combined_output(mock_run, sandbox):
    mock_run.return_value = make_proc(
        stdout="FAILED test_solution.py::test_x",
        stderr="AssertionError: assert 0 == 1",
        returncode=1,
    )
    result = sandbox.execute("x = 0", "def test_x(): assert x == 1")
    assert "FAILED" in result.output
    assert "AssertionError" in result.output


@patch("src.sandbox.subprocess.run")
def test_execute_zero_failed_in_output_still_passes(mock_run, sandbox):
    """'0 failed' in output should not trigger a false failure."""
    mock_run.return_value = make_proc("2 passed, 0 failed", returncode=0)
    result = sandbox.execute("x = 1", "def test_x(): assert x == 1")
    assert result.all_passed is True


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@patch("src.sandbox.subprocess.run")
def test_execute_timeout(mock_run, sandbox):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=30)
    result = sandbox.execute("import time; time.sleep(999)", "def test_x(): pass")
    assert result.timed_out is True
    assert result.all_passed is False
    assert result.success is False
    assert "Timed out" in result.output


# ---------------------------------------------------------------------------
# Generic exception (e.g. OSError)
# ---------------------------------------------------------------------------

@patch("src.sandbox.subprocess.run")
def test_execute_os_error(mock_run, sandbox):
    mock_run.side_effect = OSError("No such file or directory")
    result = sandbox.execute("x = 1", "def test_x(): pass")
    assert result.success is False
    assert result.all_passed is False
    assert "No such file or directory" in result.output


# ---------------------------------------------------------------------------
# subprocess invocation details
# ---------------------------------------------------------------------------

@patch("src.sandbox.subprocess.run")
def test_runs_pytest_on_test_solution(mock_run, sandbox):
    mock_run.return_value = make_proc("1 passed", returncode=0)
    sandbox.execute("x = 1", "def test_x(): assert x == 1")
    cmd = mock_run.call_args[0][0]
    assert "pytest" in " ".join(cmd)
    assert "test_solution.py" in cmd


@patch("src.sandbox.subprocess.run")
def test_preexec_fn_is_set(mock_run, sandbox):
    """Resource limits are wired in via preexec_fn."""
    mock_run.return_value = make_proc("1 passed", returncode=0)
    sandbox.execute("x = 1", "def test_x(): pass")
    kwargs = mock_run.call_args[1]
    assert kwargs.get("preexec_fn") is not None


@patch("src.sandbox.subprocess.run")
def test_timeout_kwarg_is_passed(mock_run, sandbox):
    mock_run.return_value = make_proc("1 passed", returncode=0)
    sandbox.execute("x = 1", "def test_x(): pass")
    kwargs = mock_run.call_args[1]
    assert kwargs.get("timeout") == sandbox.timeout


# ---------------------------------------------------------------------------
# Network isolation via unshare
# ---------------------------------------------------------------------------

@patch("src.sandbox._check_unshare_available", return_value=True)
@patch("src.sandbox.subprocess.run")
def test_unshare_prepended_when_available(mock_run, _mock_check):
    mock_run.return_value = make_proc("1 passed", returncode=0)
    sb = CodeSandbox()
    sb.execute("x = 1", "def test_x(): assert x == 1")
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "unshare"
    assert cmd[1] == "-n"


@patch("src.sandbox._check_unshare_available", return_value=False)
@patch("src.sandbox.subprocess.run")
def test_no_unshare_when_unavailable(mock_run, _mock_check):
    mock_run.return_value = make_proc("1 passed", returncode=0)
    sb = CodeSandbox()
    sb.execute("x = 1", "def test_x(): assert x == 1")
    cmd = mock_run.call_args[0][0]
    assert "unshare" not in cmd


# ---------------------------------------------------------------------------
# _parse_pytest_success
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("output,exit_code,expected", [
    ("1 passed", 0, True),
    ("2 passed, 0 failed", 0, True),
    ("1 failed, 1 passed", 1, False),
    ("ERROR", 1, False),
    ("1 passed", 1, False),         # exit_code trumps output text
    ("error in setup", 0, False),   # "error" in output with exit 0 → False
])
def test_parse_pytest_success(output, exit_code, expected):
    assert CodeSandbox._parse_pytest_success(output, exit_code) is expected

"""CodeSandbox: subprocess-based safe code execution.

Docker is not available on this machine. The sandbox uses subprocess.run
with resource limits applied via the POSIX resource module (RLIMIT_AS,
RLIMIT_CPU, RLIMIT_FSIZE) and an optional network namespace via
`unshare -n` when unprivileged user namespaces are enabled.
"""

import functools
import logging
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024   # 512 MB virtual address space
CPU_LIMIT_SECONDS = 30                    # wall-clock enforced by subprocess timeout;
                                          # RLIMIT_CPU adds a CPU-time hard stop
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB written files


@dataclass
class ExecutionResult:
    """Result of running code inside the sandbox.

    Attributes:
        success: True if the subprocess launched and exited normally.
        output: Combined stdout + stderr from pytest.
        all_passed: True if pytest reported 0 failures and 0 errors.
        timed_out: True if the process was killed due to timeout.
    """

    success: bool
    output: str
    all_passed: bool
    timed_out: bool = False


def _apply_resource_limits(memory_bytes: int, cpu_seconds: int, max_file_bytes: int) -> None:
    """Set POSIX resource limits in the child process (used as preexec_fn).

    Args:
        memory_bytes: Virtual address space cap (RLIMIT_AS).
        cpu_seconds: CPU-time cap in seconds (RLIMIT_CPU).
        max_file_bytes: Maximum size of any single file written (RLIMIT_FSIZE).
    """
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))


def _check_unshare_available() -> bool:
    """Return True if `unshare -n` works for network namespace isolation.

    Requires either root or unprivileged user namespaces
    (/proc/sys/kernel/unprivileged_userns_clone = 1).

    Returns:
        True if network isolation is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["unshare", "-n", "true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


class CodeSandbox:
    """Executes code and tests safely in an isolated subprocess.

    Resource limits applied per execution:
    - Virtual memory: 512 MB (RLIMIT_AS)
    - CPU time: 30 s hard stop (RLIMIT_CPU) + wall-clock timeout
    - File writes: 10 MB max per file (RLIMIT_FSIZE)
    - Network: disabled via `unshare -n` if available, otherwise unrestricted
    """

    def __init__(
        self,
        timeout: int = TIMEOUT_SECONDS,
        memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
        cpu_limit_seconds: int = CPU_LIMIT_SECONDS,
    ) -> None:
        """Initialize the CodeSandbox.

        Args:
            timeout: Wall-clock timeout in seconds for each execution.
            memory_limit_bytes: Virtual memory cap applied to the child process.
            cpu_limit_seconds: CPU-time cap applied to the child process.
        """
        self.timeout = timeout
        self.memory_limit_bytes = memory_limit_bytes
        self.cpu_limit_seconds = cpu_limit_seconds
        self._use_unshare = _check_unshare_available()

        if self._use_unshare:
            logger.info("CodeSandbox: network isolation via unshare -n")
        else:
            logger.info(
                "CodeSandbox: unshare unavailable — network isolation disabled"
            )

    def execute(self, source_code: str, test_code: str) -> ExecutionResult:
        """Run source_code against test_code in a sandboxed subprocess.

        Writes both files to a temporary directory and runs pytest against
        test_solution.py. Resource limits are applied to the child process.

        Args:
            source_code: Python source code to evaluate.
            test_code: pytest test file contents.

        Returns:
            ExecutionResult with success, output, all_passed, and timed_out.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "solution.py").write_text(source_code, encoding="utf-8")
            (tmp / "test_solution.py").write_text(test_code, encoding="utf-8")
            return self._run_pytest(tmp)

    def _run_pytest(self, workdir: Path) -> ExecutionResult:
        """Launch pytest in the sandbox subprocess.

        Args:
            workdir: Directory containing solution.py and test_solution.py.

        Returns:
            ExecutionResult.
        """
        cmd = [
            sys.executable, "-m", "pytest",
            "test_solution.py", "--tb=short", "-q", "--no-header",
        ]
        if self._use_unshare:
            cmd = ["unshare", "-n"] + cmd

        preexec = functools.partial(
            _apply_resource_limits,
            self.memory_limit_bytes,
            self.cpu_limit_seconds,
            MAX_FILE_SIZE_BYTES,
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=preexec,
            )
            output = proc.stdout + proc.stderr
            all_passed = self._parse_pytest_success(output, proc.returncode)
            return ExecutionResult(success=True, output=output, all_passed=all_passed)

        except subprocess.TimeoutExpired:
            logger.warning("Sandbox timed out after %ds", self.timeout)
            return ExecutionResult(
                success=False, output="Timed out.", all_passed=False, timed_out=True
            )

        except Exception as exc:
            logger.error("Sandbox error: %s", exc)
            return ExecutionResult(success=False, output=str(exc), all_passed=False)

    @staticmethod
    def _parse_pytest_success(output: str, exit_code: int) -> bool:
        """Determine if pytest reported all tests passing.

        Args:
            output: Combined pytest stdout + stderr.
            exit_code: Process exit code (0 = all passed in pytest).

        Returns:
            True if exit_code is 0 and no actual failures or errors appear.
        """
        if exit_code != 0:
            return False
        lower = output.lower()
        if re.search(r"\b[1-9]\d*\s+failed", lower):
            return False
        if "error" in lower:
            return False
        return True

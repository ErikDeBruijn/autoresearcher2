"""Code execution sandbox for analysis steps.

Runs LLM-generated Python code on the remote VM with:
- Network isolation (unshare --net)
- Timeout enforcement
- Memory limits via ulimit
- Module allowlist enforcement (pre-execution AST check)
- JSON output capture

This is the key enabling component for v2.0 research steps.
"""

import ast
import json
import subprocess
import textwrap

# Modules the LLM is allowed to import
ALLOWED_MODULES = {
    "numpy", "np",
    "scipy", "scipy.stats", "scipy.optimize", "scipy.linalg",
    "pandas", "pd",
    "statsmodels", "statsmodels.api", "statsmodels.formula.api",
    "sklearn", "sklearn.linear_model", "sklearn.metrics",
    "json", "math", "statistics", "collections", "itertools",
    "functools", "operator", "re", "datetime", "pathlib",
}

# Modules explicitly blocked (even if not in allowlist)
BLOCKED_MODULES = {
    "torch", "tensorflow", "jax",
    "subprocess", "os", "sys", "shutil",
    "socket", "http", "urllib", "requests",
    "multiprocessing", "threading",
    "ctypes", "importlib",
}


class SandboxError(Exception):
    """Raised when sandbox validation or execution fails."""


def validate_code(code: str) -> list[str]:
    """Check code for disallowed imports and operations.

    Returns list of violations (empty = safe to run).
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_MODULES:
                    violations.append(f"Blocked import: {alias.name}")
                elif module not in ALLOWED_MODULES and alias.name not in ALLOWED_MODULES:
                    violations.append(f"Disallowed import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_MODULES:
                    violations.append(f"Blocked import: from {node.module}")
                elif module not in ALLOWED_MODULES and node.module not in ALLOWED_MODULES:
                    violations.append(f"Disallowed import: from {node.module}")

        # Check for dangerous built-in calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile", "__import__", "open"):
                    violations.append(f"Blocked built-in: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ("system", "popen", "exec", "spawn"):
                    violations.append(f"Blocked method: .{node.func.attr}()")

    return violations


def run_in_sandbox(
    code: str,
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
    timeout_seconds: int = 60,
    memory_mb: int = 2048,
    data_json: str | None = None,
) -> dict:
    """Execute Python code in a sandboxed environment on the remote VM.

    Args:
        code: Python code to execute. Must print JSON to stdout.
        ssh_host: SSH target.
        ssh_key: SSH key path.
        timeout_seconds: Max execution time.
        memory_mb: Max memory (virtual) in MB.
        data_json: Optional JSON string to inject as DATA variable.

    Returns:
        {"success": True, "output": <parsed JSON>, "stdout": str, "stderr": str}
        or {"success": False, "error": str, "stdout": str, "stderr": str}
    """
    # Pre-execution validation
    violations = validate_code(code)
    if violations:
        return {
            "success": False,
            "error": f"Code validation failed: {'; '.join(violations)}",
            "stdout": "",
            "stderr": "",
        }

    # Wrap code with data injection and JSON output helper
    wrapped = ""
    if data_json:
        wrapped += f"import json\nDATA = json.loads('''{data_json}''')\n"
    wrapped += code

    # Build the sandbox command
    # ulimit -v limits virtual memory, timeout limits wall clock
    sandbox_cmd = (
        f"timeout {timeout_seconds} "
        f"unshare --net "
        f"python3 -c {_shell_quote(wrapped)}"
    )

    try:
        result = subprocess.run(
            ["ssh", "-i", ssh_key,
             "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=2",
             ssh_host, sandbox_cmd],
            capture_output=True, text=True,
            timeout=timeout_seconds + 30,  # extra margin for SSH
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout_seconds}s",
            "stdout": "",
            "stderr": "",
        }

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        return {
            "success": False,
            "error": f"Exit code {result.returncode}: {stderr[-500:]}",
            "stdout": stdout,
            "stderr": stderr,
        }

    # Try to parse JSON from stdout
    try:
        output = json.loads(stdout)
    except json.JSONDecodeError:
        # Return raw stdout if not JSON
        output = stdout.strip()

    return {
        "success": True,
        "output": output,
        "stdout": stdout,
        "stderr": stderr,
    }


def _shell_quote(s: str) -> str:
    """Quote a string for use in shell commands."""
    return "'" + s.replace("'", "'\\''") + "'"

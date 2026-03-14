"""Tests for code execution sandbox validation."""

from autoresearcher2.research.sandbox import (
    validate_code,
    _shell_quote,
    ALLOWED_MODULES,
    BLOCKED_MODULES,
)


class TestValidateCode:
    """Test AST-based code validation."""

    def test_safe_numpy_code(self):
        code = "import numpy as np\nresult = np.mean([1, 2, 3])"
        assert validate_code(code) == []

    def test_safe_scipy_stats(self):
        code = "from scipy.stats import pearsonr\nr, p = pearsonr([1,2,3], [4,5,6])"
        assert validate_code(code) == []

    def test_safe_pandas(self):
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1]})"
        assert validate_code(code) == []

    def test_safe_json_math(self):
        code = "import json\nimport math\nprint(json.dumps({'pi': math.pi}))"
        assert validate_code(code) == []

    def test_safe_statsmodels(self):
        code = "import statsmodels.api as sm"
        assert validate_code(code) == []

    def test_safe_sklearn(self):
        code = "from sklearn.linear_model import LinearRegression"
        assert validate_code(code) == []

    def test_blocked_subprocess(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        violations = validate_code(code)
        assert len(violations) >= 1
        assert any("subprocess" in v for v in violations)

    def test_blocked_os(self):
        code = "import os\nos.system('rm -rf /')"
        violations = validate_code(code)
        assert any("os" in v for v in violations)

    def test_blocked_sys(self):
        code = "import sys\nsys.exit(1)"
        violations = validate_code(code)
        assert any("sys" in v for v in violations)

    def test_blocked_socket(self):
        code = "import socket\ns = socket.socket()"
        violations = validate_code(code)
        assert any("socket" in v for v in violations)

    def test_blocked_requests(self):
        code = "import requests\nrequests.get('http://evil.com')"
        violations = validate_code(code)
        assert any("requests" in v for v in violations)

    def test_blocked_torch(self):
        code = "import torch"
        violations = validate_code(code)
        assert any("torch" in v for v in violations)

    def test_blocked_from_import(self):
        code = "from os import path"
        violations = validate_code(code)
        assert any("os" in v for v in violations)

    def test_blocked_from_import_http(self):
        code = "from http.server import HTTPServer"
        violations = validate_code(code)
        assert any("http" in v for v in violations)

    def test_blocked_exec(self):
        code = "exec('import os')"
        violations = validate_code(code)
        assert any("exec" in v for v in violations)

    def test_blocked_eval(self):
        code = "eval('1+1')"
        violations = validate_code(code)
        assert any("eval" in v for v in violations)

    def test_blocked_compile(self):
        code = "compile('pass', '<string>', 'exec')"
        violations = validate_code(code)
        assert any("compile" in v for v in violations)

    def test_blocked_dunder_import(self):
        code = "__import__('os')"
        violations = validate_code(code)
        assert any("__import__" in v for v in violations)

    def test_blocked_open(self):
        code = "f = open('/etc/passwd')"
        violations = validate_code(code)
        assert any("open" in v for v in violations)

    def test_blocked_system_method(self):
        code = "x.system('ls')"
        violations = validate_code(code)
        assert any(".system" in v for v in violations)

    def test_blocked_popen_method(self):
        code = "x.popen('ls')"
        violations = validate_code(code)
        assert any(".popen" in v for v in violations)

    def test_disallowed_unknown_module(self):
        code = "import pickle"
        violations = validate_code(code)
        assert any("pickle" in v for v in violations)

    def test_syntax_error(self):
        code = "def foo(:\n  pass"
        violations = validate_code(code)
        assert len(violations) == 1
        assert "Syntax error" in violations[0]

    def test_multiple_violations(self):
        code = "import os\nimport subprocess\nexec('pass')"
        violations = validate_code(code)
        assert len(violations) == 3

    def test_allowed_dotted_import(self):
        code = "import scipy.optimize"
        assert validate_code(code) == []

    def test_safe_builtins_not_blocked(self):
        """Normal function calls should not be flagged."""
        code = "print(len([1, 2, 3]))"
        assert validate_code(code) == []


class TestShellQuote:
    def test_simple_string(self):
        assert _shell_quote("hello") == "'hello'"

    def test_string_with_single_quotes(self):
        result = _shell_quote("it's")
        assert "it" in result
        assert "s" in result
        # Should be properly escaped for shell
        assert "\\'" in result

    def test_empty_string(self):
        assert _shell_quote("") == "''"

    def test_string_with_special_chars(self):
        result = _shell_quote('print("hello world")')
        assert "print" in result


class TestModuleLists:
    def test_no_overlap(self):
        """Allowed and blocked modules should not overlap."""
        overlap = ALLOWED_MODULES & BLOCKED_MODULES
        assert overlap == set(), f"Modules in both allowed and blocked: {overlap}"

    def test_key_analysis_modules_allowed(self):
        for mod in ["numpy", "scipy", "pandas", "statsmodels", "sklearn"]:
            assert mod in ALLOWED_MODULES

    def test_key_dangerous_modules_blocked(self):
        for mod in ["subprocess", "os", "sys", "socket"]:
            assert mod in BLOCKED_MODULES

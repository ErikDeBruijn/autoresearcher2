"""Reusable LLM call wrapper via claude -p over SSH.

Wraps the existing SSH pattern from llm/proposal.py into a generic
function that any v3 component can use.
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
    timeout: int = 180,
    local: bool = False,
) -> str:
    """Call claude -p, either locally or on remote VM via SSH.

    Set local=True when running on the VM itself to skip the SSH hop.
    Raises RuntimeError on failure.
    """
    if local:
        cmd = ["claude", "-p", "--output-format", "json"]
    else:
        cmd = [
            "ssh",
            "-i", ssh_key,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            ssh_host,
            "claude -p --output-format json 2>/dev/null",
        ]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{'claude' if local else 'SSH/claude'} failed (rc={result.returncode}): {result.stderr[:200]}"
        )

    return result.stdout


def call_llm_json(
    prompt: str,
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
    timeout: int = 180,
    local: bool = False,
) -> dict:
    """Call claude -p and parse JSON from response.

    Handles the claude --output-format json wrapping where the actual
    content is in {"result": "..."} format.
    """
    raw = call_llm(prompt, ssh_host, ssh_key, timeout, local=local)
    return parse_json_response(raw)


def parse_json_response(raw: str) -> dict:
    """Extract JSON from claude's response.

    Claude --output-format json wraps content in {"result": "..."}.
    The actual JSON we want may be inside that string.
    """
    raw = raw.strip()

    # Try direct parse first
    try:
        data = json.loads(raw)
        # If it's the claude --output-format json wrapper, extract the inner content
        if isinstance(data, dict) and "result" in data:
            inner = data["result"]
            if isinstance(inner, str):
                parsed = _extract_json_from_text(inner)
                if parsed is not None:
                    return parsed
                logger.warning("Could not parse inner result as JSON: %s", inner[:200])
                return data
            if isinstance(inner, dict):
                return inner
        # If no result key, return as-is (direct JSON response)
        return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown
    for marker in ["```json", "```"]:
        if marker in raw:
            start = raw.index(marker) + len(marker)
            end = raw.index("```", start)
            try:
                return json.loads(raw[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    raise ValueError(f"Could not parse JSON from response: {raw[:200]}")


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from text that may contain markdown fences, preamble, or trailing text."""
    text = text.strip()

    # Strategy 1: direct parse (clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code fences
    import re
    fence_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { ... } block (greedy from first { to last })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None

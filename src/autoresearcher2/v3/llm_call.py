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
) -> str:
    """Call claude -p on remote VM, return raw text response.

    Raises RuntimeError on SSH or claude failure.
    """
    ssh_cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        ssh_host,
        "claude -p --output-format json 2>/dev/null",
    ]

    result = subprocess.run(
        ssh_cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"SSH/claude failed (rc={result.returncode}): {result.stderr[:200]}"
        )

    return result.stdout


def call_llm_json(
    prompt: str,
    ssh_host: str = "root@dllm-experiment.home",
    ssh_key: str = "~/.ssh/pve03_key",
    timeout: int = 180,
) -> dict:
    """Call claude -p and parse JSON from response.

    Handles the claude --output-format json wrapping where the actual
    content is in {"result": "..."} format.
    """
    raw = call_llm(prompt, ssh_host, ssh_key, timeout)
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
                # Try to parse the inner content as JSON
                inner = inner.strip()
                # Strip markdown code fences if present
                if inner.startswith("```json"):
                    inner = inner[7:]
                if inner.startswith("```"):
                    inner = inner[3:]
                if inner.endswith("```"):
                    inner = inner[:-3]
                inner = inner.strip()
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
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

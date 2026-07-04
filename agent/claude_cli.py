"""
Claude Code CLI provider: run LLM calls through `claude -p` (headless print
mode) instead of the Anthropic API. Uses the local Claude Code install's
authentication, so no ANTHROPIC_API_KEY is needed.

All tools are disabled and session persistence is off -- each call is a
single stateless completion, like a messages.create() with a system prompt.
"""

import json
import shutil
import subprocess


class ClaudeCLIError(RuntimeError):
    """The claude CLI failed or returned an error result."""


def claude_cli_available() -> bool:
    """True if the `claude` binary is on PATH."""
    return shutil.which("claude") is not None


def run_claude_cli(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    timeout: float = 300.0,
) -> str:
    """
    Run a single stateless completion through `claude -p`.

    Args:
        system_prompt: System prompt for the call
        user_message: User message (passed via stdin)
        model: Model to use; None uses the CLI's configured default
        timeout: Seconds to wait before giving up

    Returns:
        The model's response text.
    """
    if not claude_cli_available():
        raise ClaudeCLIError(
            "`claude` CLI not found on PATH. Install Claude Code or use "
            "--provider anthropic with an ANTHROPIC_API_KEY."
        )

    cmd = [
        "claude", "-p",
        "--tools", "",                 # pure completion, no tool use
        "--no-session-persistence",
        "--output-format", "json",
        "--system-prompt", system_prompt,
    ]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd, input=user_message,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude -p timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude -p exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCLIError(f"claude -p returned non-JSON output: {proc.stdout[:500]}") from e

    if envelope.get("is_error"):
        raise ClaudeCLIError(f"claude -p error result: {envelope.get('result')}")

    result = envelope.get("result")
    if not isinstance(result, str):
        raise ClaudeCLIError(f"claude -p envelope missing result text: {envelope}")
    return result

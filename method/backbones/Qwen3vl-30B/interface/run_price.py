#!/usr/bin/python3
"""Single-agent_price runner aligned with the ImageReward API evaluation flow."""

from pathlib import Path
import sys

INTERFACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTERFACE_DIR.parents[2]
GEMINI_RUNNER = PROJECT_ROOT / "method" / "Gemini-3-flash" / "interface" / "run_price.py"

if __name__ == "__main__":
    code = compile(GEMINI_RUNNER.read_text(encoding="utf-8"), str(GEMINI_RUNNER), "exec")
    globals_dict = {"__name__": "__main__", "__file__": str(GEMINI_RUNNER)}
    exec(code, globals_dict)

"""Backward-compatible entrypoint for GPT-5-mini preset.

All shared logic lives in main_simulation.py. Keep this file so existing
commands/scripts still work.
"""

from main_simulation import PRESETS, run_simulation


def main() -> None:
    run_simulation(PRESETS["gpt_5_mini"])


if __name__ == "__main__":
    main()
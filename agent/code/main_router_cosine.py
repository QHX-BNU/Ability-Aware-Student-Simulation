"""Backward-compatible entrypoint for router_cosine preset.

The per-student (model, base_url, api_key) routing logic is implemented in
main_simulation.py via make_router_llm_resolver().
"""

from main_simulation import PRESETS, run_simulation


def main() -> None:
    run_simulation(PRESETS["router_cosine"])


if __name__ == "__main__":
    main()


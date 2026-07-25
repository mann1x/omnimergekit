"""single-turn-seed-harness -- reproduce the June single-turn HumanEval+ /
MultiPL-E seed sweep that isolated the Gemma-4 chat-template reasoning
re-injection failure.

For each (problem x seed) it sends ONE single-turn chat request to a running
llama-server and grades the response with the hammer classifier (copied verbatim
from opencode_capture/hammer_raw.py): CLEAN / RUNAWAY / THINK_EXPLODE / CORRUPT /
ABORT. The chat template under test is chosen by which server you point at, so an
OLD-vs-NEW template comparison is two runs against two servers.

Public API:
    from single_turn_seed_harness.classify import classify
    from single_turn_seed_harness.tasks import load_tasks
    from single_turn_seed_harness.sweep import run_sweep
    from single_turn_seed_harness.cli import main
"""
__version__ = "0.1.0"

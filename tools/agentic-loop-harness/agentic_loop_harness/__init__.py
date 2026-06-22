"""agentic-loop-harness -- a self-contained harness that measures how often a
chat-served LLM falls into a degenerate agentic loop, isolating the effect of the
chat template, sampler, and reasoning settings.

Public API:
    from agentic_loop_harness.detect import detect_turn_loop
    from agentic_loop_harness.replay import replay_fixture, chat
    from agentic_loop_harness.server import LlamaServer
    from agentic_loop_harness.cli import run, main
"""
__version__ = "0.1.0"

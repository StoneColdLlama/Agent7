"""
Agent7.py — Terminal Interface
-------------------------------
Thin terminal wrapper around agent_core.py.
All agent logic lives in agent_core — this file only handles the
terminal-specific concerns: the spinner, the confirm prompt for bash
commands, the CLI argument parser, and the REPL loop.

Usage:
    python Agent7.py
    python Agent7.py --model qwen3-coder:30b
    python Agent7.py --model qwen2.5-coder:7b --max-steps 10 --ctx 16384
    python Agent7.py --no-thinking
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import argparse
import itertools as _itertools
import sys
import threading
import time

from agent_core import (
    # Constants
    DEFAULT_MODEL, DEFAULT_MAX_STEPS, DEFAULT_CTX,
    MEMORY_FILE, MEMORY_SESSIONS_TO_INJECT,
    # Memory
    load_memory, save_memory, format_memory_for_prompt,
    SessionRecorder,
    # Agent
    ThinkingModel, BaseBashTool, build_model, build_agent,
    build_system_prompt,
)

# Knowledge base — optional
try:
    from knowledge import (
        load_facts, handle_knowledge_command, explore_topic,
        check_and_flag_gap, find_relevant_facts,
        build_facts_prompt_block, FACTS_FILE,
    )
    HAS_KNOWLEDGE = True
except ImportError:
    HAS_KNOWLEDGE = False

from smolagents import LiteLLMModel


# ─────────────────────────────────────────────
# SPINNER
# Animated terminal indicator shown while the agent is working.
# Runs in a background thread so it doesn't block the REPL.
# ─────────────────────────────────────────────

class Spinner:
    """Animated terminal spinner that runs in a background thread."""

    def __init__(self, message: str = "Working"):
        self.message     = message
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for frame in _itertools.cycle(frames):
            if self._stop_event.is_set():
                break
            print(f"\r\033[36m{frame} {self.message}...\033[0m", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (len(self.message) + 15) + "\r", end="", flush=True)

    def start(self): self._thread.start()
    def stop(self):  self._stop_event.set(); self._thread.join()


# ─────────────────────────────────────────────
# TERMINAL THINKING MODEL
# Overrides on_thinking() to print <think> blocks in dim cyan.
# When --no-thinking is passed, uses plain LiteLLMModel instead.
# ─────────────────────────────────────────────

class TerminalThinkingModel(ThinkingModel):
    """Prints chain-of-thought blocks in dim cyan in the terminal."""

    def on_thinking(self, thought: str) -> None:
        print(f"\n\033[2m\033[36m💭 Thinking:\n{thought}\033[0m\n")


# ─────────────────────────────────────────────
# TERMINAL BASH TOOL
# Asks for user confirmation before running any shell command.
# ─────────────────────────────────────────────

class TerminalBashTool(BaseBashTool):
    """Prompts the user before executing bash commands."""

    def on_command(self, command: str) -> None:
        print(f"\n[BashTool] Command to run:\n  {command}")

    def confirm(self, command: str) -> bool:
        answer = input("  Execute? [y/N]: ").strip().lower()
        return answer == "y"


# ─────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Agent7 — Local Coding Agent (Terminal)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-steps",  type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max reasoning steps (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--ctx",        type=int, default=DEFAULT_CTX,
                        help=f"Context window tokens (default: {DEFAULT_CTX})")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable chain-of-thought display")
    return parser.parse_args()


# ─────────────────────────────────────────────
# SHUTDOWN HELPER
# ─────────────────────────────────────────────

def _shutdown(recorder: SessionRecorder, all_sessions: list) -> None:
    """Save session memory and exit."""
    if recorder.prompts:
        all_sessions.append(recorder.to_session_dict())
        save_memory(all_sessions)
        print(f"\n[Memory] Session saved ({len(all_sessions)} total sessions)")
    else:
        print("\n[Memory] No tasks this session — nothing saved.")
    print("Goodbye!")
    sys.exit(0)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args         = parse_args()
    show_thinking = not args.no_thinking

    # --- Load memory ---
    all_sessions  = load_memory()
    session_count = len(all_sessions)

    # --- Load facts ---
    all_facts = load_facts(FACTS_FILE) if HAS_KNOWLEDGE else []

    print(f"\n🤖 Agent7 — Local Coding Agent")
    print(f"   Model      : {args.model}")
    print(f"   Max steps  : {args.max_steps}")
    print(f"   Context    : {args.ctx} tokens")
    print(f"   Thinking   : {'on' if show_thinking else 'off'}")
    print(f"   Memory     : {session_count} past session(s)")
    print(f"   Facts      : {len(all_facts)} in knowledge base")
    print(f"\n   Type 'quit' or 'exit' to stop.")
    if HAS_KNOWLEDGE:
        print(f"   Type 'explore: <topic>' to research and save a fact.")
        print(f"   Type 'facts list' to view your knowledge base.")
    print()

    # --- Build model ---
    model_class = TerminalThinkingModel if show_thinking else LiteLLMModel
    model       = build_model(args.model, args.ctx, model_class=model_class)

    # --- Build model_fn for knowledge discussions ---
    def model_fn(messages):
        response = model(messages)
        return response.content if hasattr(response, "content") else str(response)

    def search_fn(query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = [f"- {r['title']}: {r['body']}"
                           for r in ddgs.text(query, max_results=5)]
            return "\n".join(results) or "No results found."
        except Exception as e:
            return f"Search unavailable: {e}"

    # --- Build agent ---
    agent = build_agent(model, TerminalBashTool(), args.max_steps)

    # --- Build memory + facts prefix ---
    system_prompt = build_system_prompt(all_sessions, MEMORY_SESSIONS_TO_INJECT)
    facts_block   = build_facts_prompt_block(all_facts) if HAS_KNOWLEDGE else ""
    prefix_parts  = []
    if system_prompt and all_sessions:
        prefix_parts.append(system_prompt)
    if facts_block:
        prefix_parts.append(facts_block)
    memory_prefix = "\n\n".join(prefix_parts)

    # --- Session recorder ---
    recorder = SessionRecorder()

    # --- REPL loop ---
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _shutdown(recorder, all_sessions)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            _shutdown(recorder, all_sessions)

        # Knowledge management commands (facts list, facts search, etc.)
        if HAS_KNOWLEDGE and handle_knowledge_command(user_input, all_facts):
            continue

        # Manual exploration
        if HAS_KNOWLEDGE and user_input.lower().startswith("explore:"):
            topic = user_input[len("explore:"):].strip()
            if topic:
                explore_topic(topic, model_fn, search_fn, all_facts)
            else:
                print("Usage: explore: <topic>\n")
            continue

        # Show relevant facts before running agent
        if HAS_KNOWLEDGE and all_facts:
            relevant = find_relevant_facts(user_input, all_facts, max_results=3)
            if relevant:
                print("\n📚 Relevant fact(s) from knowledge base:")
                for score, f in relevant:
                    print(f"   [#{f['id']}] {f['topic']}: {f['fact'][:150]}")
                print()

        # Run the agent
        try:
            if memory_prefix:
                full_input    = memory_prefix + "\n\nCurrent task: " + user_input
                memory_prefix = ""
            else:
                full_input = user_input

            spinner = Spinner("Agent is thinking")
            spinner.start()
            try:
                response = agent.run(full_input)
            finally:
                spinner.stop()

            print(f"\nAgent: {response}\n")
            recorder.record(user_input, response)

            # Auto gap detection
            if HAS_KNOWLEDGE:
                check_and_flag_gap(user_input, str(response), model_fn, all_facts)

        except KeyboardInterrupt:
            print("\n[Task interrupted]\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()

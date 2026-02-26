"""
agent_core.py — Shared Agent Core
----------------------------------
Single source of truth for everything both Agent7.py (terminal) and
web_agent.py (web UI) need. Neither interface defines agent logic —
they just call functions from here.

What lives here:
  - User constants (model, steps, context window, file paths)
  - Memory system  (load / save / format sessions)
  - Facts system   (load facts, find relevant facts)
  - SessionRecorder
  - ThinkingModel  (LiteLLMModel subclass that extracts <think> blocks)
  - BashTool base  (shared logic; each interface subclasses for confirm behaviour)
  - build_agent()  (creates and returns a configured CodeAgent)
  - build_system_prompt()
"""

# ─────────────────────────────────────────────
# USER CONSTANTS — edit these to customise behaviour
# ─────────────────────────────────────────────

# Default Ollama model if none is passed via --model flag.
DEFAULT_MODEL = "qwen2.5-coder:14b"

# Default maximum reasoning steps per task.
DEFAULT_MAX_STEPS = 15

# Default context window size in tokens.
DEFAULT_CTX = 8192

# How many past sessions to inject into the system prompt.
MEMORY_SESSIONS_TO_INJECT = 5

# Path to the session memory file.
MEMORY_FILE = "memory.json"

# Path to the knowledge base facts file.
FACTS_FILE = "facts.json"

# Directory where agent-created files are copied for web download.
OUTPUT_DIR = "agent_outputs"


# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, Tool

# Knowledge base — optional, gracefully skipped if not present
try:
    from knowledge import load_facts, find_relevant_facts, FACTS_FILE as KB_FACTS_FILE
    HAS_KNOWLEDGE = True
except ImportError:
    HAS_KNOWLEDGE = False


# ─────────────────────────────────────────────
# MEMORY SYSTEM
# Handles loading, saving, and formatting of past session memory.
# Each session is stored as a dict:
#   { "date": str, "prompts": [str], "summary": str }
# ─────────────────────────────────────────────

def load_memory(path: str = MEMORY_FILE) -> list:
    """Load sessions from JSON file. Returns empty list if file doesn't exist."""
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            print(f"[Memory] Warning: could not load memory file: {e}")
    return []


def save_memory(sessions: list, path: str = MEMORY_FILE) -> None:
    """Save all session records to JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        print(f"[Memory] Warning: could not save memory: {e}")


def format_memory_for_prompt(sessions: list, n: int = MEMORY_SESSIONS_TO_INJECT) -> str:
    """
    Format the last N sessions into a string ready for injection into
    the agent's first prompt. Returns empty string if no sessions.
    """
    if not sessions:
        return ""

    recent = sessions[-n:]
    lines = ["## Your memory from previous sessions:\n"]
    for i, session in enumerate(recent):
        date    = session.get("date", "unknown date")
        prompts = session.get("prompts", [])
        summary = session.get("summary", "")
        lines.append(f"### Session {i + 1} — {date}")
        if prompts:
            lines.append(f"Tasks worked on: {'; '.join(prompts[:5])}")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SESSION RECORDER
# Tracks what the user asked during the current run so it can be saved.
# ─────────────────────────────────────────────

class SessionRecorder:
    """Records prompts and results from the current session."""

    def __init__(self):
        self.prompts    = []
        self.results    = []
        self.start_time = datetime.now()

    def record(self, prompt: str, result: str) -> None:
        self.prompts.append(prompt.strip())
        self.results.append(str(result).strip()[:500])

    def to_session_dict(self) -> dict:
        summary = f"Completed {len(self.prompts)} task(s)."
        if self.prompts:
            summary += f" Last task: {self.prompts[-1][:120]}"
        return {
            "date":    self.start_time.strftime("%Y-%m-%d %H:%M"),
            "prompts": self.prompts,
            "summary": summary,
        }


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

# Rules injected into every session so the agent always saves files.
FILE_SAVING_RULES = (
    "FILE SAVING RULES — always follow these:\n"
    "- When you generate any code (Python, bash, HTML, etc.), ALWAYS save it to a "
    "file using the python_interpreter tool with open('filename', 'w'). "
    "Never just print code.\n"
    "- When asked to create a spreadsheet, use openpyxl to write a .xlsx file.\n"
    "- When asked to create a Word document, use python-docx to write a .docx file.\n"
    "- When asked to create a PDF, use fpdf2 or reportlab to write a .pdf file.\n"
    "- Always tell the user the full path of every file you save.\n"
    "- If the user does not specify a filename, invent a sensible one.\n\n"
)


def build_system_prompt(sessions: list, n: int = MEMORY_SESSIONS_TO_INJECT) -> str:
    """
    Build the full system prompt including file-saving rules and
    injected memory from past sessions.
    """
    base = (
        "You are a powerful local coding agent running on the user's machine. "
        "You can write and execute Python code, run bash commands (with user approval), "
        "and search the web. Be concise, practical, and always explain what you are doing.\n\n"
        + FILE_SAVING_RULES
    )

    memory_block = format_memory_for_prompt(sessions, n)
    if memory_block:
        base += memory_block
        base += (
            "\nUse this memory to provide continuity — reference past work when relevant, "
            "avoid repeating mistakes, and build on what was previously accomplished.\n"
        )

    return base


# ─────────────────────────────────────────────
# THINKING MODEL
# LiteLLMModel subclass that extracts <think>...</think> blocks from
# reasoning models (qwen3, deepseek-r1). Each interface decides what
# to do with the extracted thought — terminal prints it in dim cyan,
# web forwards it as a collapsible block.
# ─────────────────────────────────────────────

class ThinkingModel(LiteLLMModel):
    """
    Intercepts model responses and extracts <think> blocks.
    Calls self.on_thinking(thought_text) for each block found.
    Subclass and override on_thinking() to customise behaviour.
    """

    def on_thinking(self, thought: str) -> None:
        """Called for each <think> block. Override in subclasses."""
        # Default: print in dim cyan (terminal behaviour)
        print(f"\n\033[2m\033[36m💭 Thinking:\n{thought}\033[0m\n")

    def __call__(self, messages, **kwargs):
        response = super().__call__(messages, **kwargs)
        try:
            raw = response.raw
            if isinstance(raw, dict):
                choices = raw.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "") or ""
                    thoughts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
                    for thought in thoughts:
                        thought = thought.strip()
                        if thought:
                            self.on_thinking(thought)
        except Exception:
            pass
        return response


# ─────────────────────────────────────────────
# BASH TOOL BASE
# Shared bash execution logic. The confirmation behaviour differs
# between interfaces:
#   - Terminal (Agent7.py): asks "Execute? [y/N]" in the terminal
#   - Web (web_agent.py):   executes automatically, shows command in UI
# Each interface subclasses BaseBashTool and overrides confirm().
# ─────────────────────────────────────────────

class BaseBashTool(Tool):
    name = "bash"
    description = (
        "Runs a bash command on the local machine and returns stdout + stderr. "
        "Use for file operations, running scripts, installing packages, "
        "checking system state, or any shell task. "
        "Commands run in the current working directory."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "The bash command to execute.",
        }
    }
    output_type = "string"

    def confirm(self, command: str) -> bool:
        """
        Return True to allow execution, False to cancel.
        Override in subclasses to change confirmation behaviour.
        """
        return True  # default: allow (web mode)

    def on_command(self, command: str) -> None:
        """Called just before a command runs. Override to display/log it."""
        pass

    def forward(self, command: str) -> str:
        self.on_command(command)
        if not self.confirm(command):
            return "Command cancelled."
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=120,
            )
            output = result.stdout or ""
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120 seconds."
        except Exception as e:
            return f"Error: {e}"


# ─────────────────────────────────────────────
# AGENT FACTORY
# Creates and returns a fully configured CodeAgent.
# Both interfaces call this instead of duplicating setup code.
# ─────────────────────────────────────────────

def build_agent(
    model: LiteLLMModel,
    bash_tool: BaseBashTool = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> CodeAgent:
    """
    Build and return a CodeAgent with the standard tool set.

    Arguments:
        model      — a ThinkingModel (or subclass) instance
        bash_tool  — a BaseBashTool subclass instance; if None, uses base
        max_steps  — max reasoning steps per task
    """
    if bash_tool is None:
        bash_tool = BaseBashTool()

    return CodeAgent(
        tools=[bash_tool, DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,          # adds built-in Python interpreter
        max_steps=max_steps,
        verbosity_level=1,
        additional_authorized_imports=["*"],  # allow any installed package
    )


def build_model(
    model_name: str = DEFAULT_MODEL,
    ctx: int = DEFAULT_CTX,
    model_class=None,
) -> LiteLLMModel:
    """
    Build and return a model instance.

    Arguments:
        model_name  — Ollama model name e.g. "qwen3-coder:30b"
        ctx         — context window size in tokens
        model_class — ThinkingModel subclass to use; defaults to ThinkingModel
    """
    if model_class is None:
        model_class = ThinkingModel

    return model_class(
        model_id=f"ollama_chat/{model_name}",
        api_base="http://127.0.0.1:11434",
        num_ctx=ctx,
    )

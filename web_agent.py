"""
web_agent.py — Web Interface
------------------------------
Thin FastAPI wrapper around agent_core.py.
All agent logic lives in agent_core — this file only handles the
web-specific concerns: HTTP routes, SSE streaming, file serving,
output capture from background threads, and the browser API.

Usage:
    pip install fastapi uvicorn sse-starlette
    python web_agent.py
    python web_agent.py --model qwen3-coder:30b --port 8080

Then open http://localhost:7860 in your browser.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import argparse
import asyncio
import json
import logging
import queue
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from agent_core import (
    # Constants
    DEFAULT_MODEL, DEFAULT_MAX_STEPS, DEFAULT_CTX,
    MEMORY_FILE, MEMORY_SESSIONS_TO_INJECT, OUTPUT_DIR,
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
        load_facts, find_relevant_facts, FACTS_FILE,
    )
    HAS_KNOWLEDGE = True
except ImportError:
    HAS_KNOWLEDGE = False
    FACTS_FILE = "facts.json"


# ─────────────────────────────────────────────
# OUTPUT QUEUE
# The agent runs in a background thread. This queue is the pipe that
# carries output events from the agent thread to the SSE stream
# which sends them to the browser in real time.
# ─────────────────────────────────────────────

output_queue = queue.Queue()
agent_lock   = threading.Lock()   # prevents concurrent agent runs
stop_requested = threading.Event()  # set this to interrupt the agent


# ─────────────────────────────────────────────
# OUTPUT CAPTURE — TeeStream
# smolagents writes Thought/Code/Observation blocks to stdout and stderr.
# TeeStream wraps both streams so every line is forwarded into the queue.
# It's thread-safe (per-thread buffers) and echoes to the real terminal too.
# ─────────────────────────────────────────────

_ANSI = re.compile(r'\x1b\[[0-9;]*m')

_SKIP_PATTERNS = [
    "POST Request Sent", "curl -X POST", "LiteLLM completion()",
    "http://127.0.0.1", "'stream'", "RAW RESPONSE", "logging_obj",
    "api_base", "num_ctx", "litellm.utils",
]

def _should_skip(text: str) -> bool:
    return any(p in text for p in _SKIP_PATTERNS)


class TeeStream:
    """
    Wraps stdout or stderr globally so all threads write through it.
    Forwards each complete line into output_queue for the SSE stream
    while still echoing to the real terminal.
    Uses per-thread buffers so concurrent writes don't corrupt each other.
    """

    def __init__(self, real_stream):
        self._real = real_stream
        self._buf  = {}              # { thread_id: str }
        self._lock = threading.Lock()

    def write(self, text):
        self._real.write(text)
        tid = threading.get_ident()
        with self._lock:
            self._buf.setdefault(tid, "")
            self._buf[tid] += text
            while "\n" in self._buf[tid]:
                line, self._buf[tid] = self._buf[tid].split("\n", 1)
                clean = _ANSI.sub('', line).strip()
                if clean and not _should_skip(clean):
                    output_queue.put({"type": "output", "text": clean})

    def flush(self):       self._real.flush()
    def fileno(self):      return self._real.fileno()
    def __getattr__(self, name): return getattr(self._real, name)


def attach_output_capture():
    """Install TeeStream on stdout and stderr, silence noisy loggers."""
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.stderr)
    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.stdout)
    for name in ("litellm", "LiteLLM", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


# ─────────────────────────────────────────────
# WEB THINKING MODEL
# Overrides on_thinking() to send <think> blocks to the browser
# as "thinking" events instead of printing them to the terminal.
# ─────────────────────────────────────────────

class WebThinkingModel(ThinkingModel):
    """Forwards <think> blocks into the output queue as thinking events.
    Also checks the stop flag before each model call so the agent halts
    cleanly between steps when the user presses Stop.
    """

    def on_thinking(self, thought: str) -> None:
        output_queue.put({"type": "thinking", "text": thought})

    def __call__(self, messages, **kwargs):
        if stop_requested.is_set():
            raise InterruptedError("Stopped by user")
        return super().__call__(messages, **kwargs)


# ─────────────────────────────────────────────
# WEB BASH TOOL
# Executes commands automatically (no terminal prompt in web mode).
# Shows the command in the browser output panel before running.
# Also triggers a file scan after each command so new files appear
# immediately in the Files tab.
# ─────────────────────────────────────────────

class WebBashTool(BaseBashTool):
    """Web version — auto-executes and forwards command to the output queue."""

    def on_command(self, command: str) -> None:
        output_queue.put({"type": "bash", "text": command})

    def confirm(self, command: str) -> bool:
        return True   # auto-approve in web mode

    def forward(self, command: str) -> str:
        result = super().forward(command)
        _scan_for_new_files()   # check for new files after every bash call
        return result


# ─────────────────────────────────────────────
# FILE TRACKING
# Watches the working directory for files created during a session
# and makes them available as download links in the Files tab.
# ─────────────────────────────────────────────

_known_files:   set  = set()
_session_files: list = []



def _init_file_tracking():
    """Record files already in agent_outputs/ before the session starts."""
    global _known_files
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    _known_files = set(str(p) for p in out.rglob('*') if p.is_file())


def _make_session_dir(prompt: str) -> str:
    """
    Create a timestamped subfolder inside agent_outputs/ for this prompt.
    Folder name: YYYY-MM-DD_HH-MM-SS_first-few-words
    Returns the folder path as a string.
    """
    import re as _re
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    words = _re.sub(r"[^a-zA-Z0-9 ]", "", prompt).split()[:4]
    slug  = "_".join(w.lower() for w in words) if words else "task"
    folder = Path(OUTPUT_DIR) / f"{timestamp}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _scan_for_new_files():
    """Check agent_outputs/ for new files and register them for the Files tab."""
    global _known_files, _session_files
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    for p in out.iterdir():
        if not p.is_file():
            continue
        path_str = str(p)
        if path_str not in _known_files:
            try:
                entry = {
                    "name": p.name,
                    "path": path_str,
                    "size": _fmt_size(p.stat().st_size),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "folder": Path(current_session_dir).name,
                }
                _session_files.append(entry)
                output_queue.put({"type": "file", "file": entry})
                _known_files.add(path_str)
            except Exception:
                pass


def _fmt_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ─────────────────────────────────────────────
# GLOBAL AGENT STATE
# One agent + model instance shared across all requests.
# ─────────────────────────────────────────────

agent_instance   = None
model_instance   = None
all_sessions:    list = []
all_facts:       list = []
session_recorder = None
memory_prefix    = ""


def setup(model_name: str, max_steps: int, ctx: int):
    """Initialise everything. Called once at server startup."""
    global agent_instance, model_instance
    global all_sessions, all_facts, session_recorder, memory_prefix

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    all_sessions     = load_memory()
    all_facts        = load_facts(FACTS_FILE) if HAS_KNOWLEDGE else []
    session_recorder = SessionRecorder()

    memory_prefix = (
        build_system_prompt(all_sessions, MEMORY_SESSIONS_TO_INJECT)
        if all_sessions else ""
    )

    model_instance = build_model(model_name, ctx, model_class=WebThinkingModel)
    agent_instance = build_agent(model_instance, WebBashTool(), max_steps)

    _init_file_tracking()
    attach_output_capture()
    print(f"[web_agent] Ready — model: {model_name}", flush=True)


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(title="Agent7 Web UI")
templates = Jinja2Templates(directory="templates")
app.mount("/static",    StaticFiles(directory="static"),    name="static")
app.mount("/downloads", StaticFiles(directory=OUTPUT_DIR),  name="downloads")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def api_status():
    return {
        "model":    app.state.model_name,
        "sessions": len(all_sessions),
        "facts":    len(all_facts),
        "files":    len(_session_files),
        "busy":     agent_lock.locked(),
    }


@app.get("/api/memory")
async def api_memory():
    return {"sessions": all_sessions[-10:]}


@app.get("/api/facts")
async def api_facts():
    facts = load_facts(FACTS_FILE) if HAS_KNOWLEDGE else []
    return {"facts": facts}


@app.get("/api/files")
async def api_files():
    return {"files": _session_files}


@app.post("/api/stop")
async def api_stop():
    """Signal the agent to stop after its current step completes."""
    stop_requested.set()
    return {"status": "stop requested"}


@app.post("/api/chat")
async def api_chat(request: Request):
    """Receive a user message and start the agent in a background thread."""
    global memory_prefix

    body       = await request.json()
    user_input = body.get("message", "").strip()

    if not user_input:
        return {"error": "empty message"}
    if agent_lock.locked():
        return {"error": "Agent is busy — please wait."}

    # Drain stale queue items
    while not output_queue.empty():
        try: output_queue.get_nowait()
        except queue.Empty: break

    def run_agent():
        global memory_prefix
        with agent_lock:
            try:
                global current_session_dir
                stop_requested.clear()
                current_session_dir = _make_session_dir(user_input)
                output_queue.put({"type": "start"})
                output_queue.put({"type": "session_dir", "folder": Path(current_session_dir).name})

                # Show relevant facts in the chat panel
                if HAS_KNOWLEDGE and all_facts:
                    relevant = find_relevant_facts(user_input, all_facts, max_results=3)
                    if relevant:
                        facts_text = "\n".join(
                            f"[#{f['id']}] {f['topic']}: {f['fact']}"
                            for _, f in relevant
                        )
                        output_queue.put({"type": "facts_match", "text": facts_text})

                # Build prompt — inject session dir so agent saves files there
                session_instruction = (
                    f"Save ALL files into this folder: {current_session_dir}\n"
                    f"Use open(\"{current_session_dir}/filename\", \"w\") to save any file.\n\n"
                )
                if memory_prefix:
                    full_input    = memory_prefix + "\n\n" + session_instruction + "Current task: " + user_input
                    memory_prefix = ""
                else:
                    full_input = session_instruction + user_input

                try:
                    response = agent_instance.run(full_input)
                except InterruptedError:
                    output_queue.put({"type": "done", "text": "⛔ Agent stopped by user."})
                    return

                if session_recorder:
                    session_recorder.record(user_input, str(response))

                _scan_for_new_files()
                output_queue.put({"type": "done", "text": str(response)})

            except Exception as e:
                output_queue.put({"type": "error", "text": str(e)})

    threading.Thread(target=run_agent, daemon=True).start()
    return {"status": "started"}


@app.get("/api/stream")
async def api_stream(request: Request):
    """SSE endpoint — browser connects here for real-time agent output."""

    async def generator() -> AsyncGenerator:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = output_queue.get(timeout=0.1)
                yield {"data": json.dumps(item)}
                if item.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield {"comment": "keepalive"}
                await asyncio.sleep(0.05)

    return EventSourceResponse(generator())


@app.on_event("shutdown")
def on_shutdown():
    if session_recorder and session_recorder.prompts:
        all_sessions.append(session_recorder.to_session_dict())
        save_memory(all_sessions)
        print(f"[web_agent] Session saved ({len(all_sessions)} total)")


# ─────────────────────────────────────────────
# CLI + ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Agent7 Web UI")
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--ctx",       type=int, default=DEFAULT_CTX)
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--host",      default="127.0.0.1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.state.model_name = args.model
    setup(args.model, args.max_steps, args.ctx)
    print(f"\n🌐 Agent7 Web UI")
    print(f"   Open http://{args.host}:{args.port} in your browser\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

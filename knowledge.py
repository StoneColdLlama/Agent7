"""
knowledge.py — Self-Teaching Knowledge Base
--------------------------------------------
Gives the agent the ability to explore topics, discuss findings with you,
and grow a persistent facts.json knowledge base over time.

How it works:
  - Manual exploration : type  explore: <topic>  in the agent
  - Automatic          : agent flags gaps during normal tasks
  - Discussion         : agent presents finding, you guide it, fact is saved
  - Fact checking      : agent checks facts.json before generating new answers

This file is imported by agentMem.py — do not run it directly.

Facts are stored in facts.json as free-form notes the agent writes itself.
"""

# ─────────────────────────────────────────────
# USER CONSTANTS
# ─────────────────────────────────────────────

# Path to the facts knowledge base file.
FACTS_FILE = "facts.json"

# How many facts to inject into context when answering questions.
# Higher = more context used. Lower = faster but less fact-aware.
FACTS_TO_INJECT = 20

# How similar a query must be to a fact topic to count as a match (0.0–1.0).
# Lower = more lenient matching. Higher = stricter.
SIMILARITY_THRESHOLD = 0.3


# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# FACTS STORAGE
# Each fact is a dict:
#   {
#     "id":        int        — unique incrementing ID
#     "topic":     str        — short topic label the agent assigns
#     "finding":   str        — what the agent originally found/believed
#     "fact":      str        — the agreed correct answer after discussion
#     "date":      str        — ISO timestamp when fact was committed
#     "source":    str        — "explored" | "automatic" | "manual"
#   }
# ─────────────────────────────────────────────

def load_facts(path: str = FACTS_FILE) -> list:
    """Load all facts from JSON file. Returns empty list if not found."""
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            print(f"[Knowledge] Warning: could not load facts: {e}")
    return []


def save_facts(facts: list, path: str = FACTS_FILE) -> None:
    """Save all facts to JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(facts, f, indent=2)
    except Exception as e:
        print(f"[Knowledge] Warning: could not save facts: {e}")


def add_fact(facts: list, topic: str, finding: str, agreed_fact: str,
             source: str = "explored") -> dict:
    """
    Create a new fact dict, append it to the facts list, save to disk,
    and return the new fact.
    """
    new_id = max((f["id"] for f in facts), default=0) + 1
    fact = {
        "id":      new_id,
        "topic":   topic.strip(),
        "finding": finding.strip(),
        "fact":    agreed_fact.strip(),
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source":  source,
    }
    facts.append(fact)
    save_facts(facts)
    return fact


def update_fact(facts: list, fact_id: int, new_fact_text: str) -> bool:
    """Update an existing fact by ID. Returns True if found and updated."""
    for f in facts:
        if f["id"] == fact_id:
            f["fact"] = new_fact_text.strip()
            f["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            save_facts(facts)
            return True
    return False


def delete_fact(facts: list, fact_id: int) -> bool:
    """Delete a fact by ID. Returns True if found and deleted."""
    original_len = len(facts)
    facts[:] = [f for f in facts if f["id"] != fact_id]
    if len(facts) < original_len:
        save_facts(facts)
        return True
    return False


# ─────────────────────────────────────────────
# FACT LOOKUP
# Simple keyword-based similarity search so the agent can check its
# knowledge base before generating a new answer.
# ─────────────────────────────────────────────

def _tokenise(text: str) -> set:
    """Split text into lowercase words, stripping punctuation."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def find_relevant_facts(query: str, facts: list,
                        threshold: float = SIMILARITY_THRESHOLD,
                        max_results: int = 5) -> list:
    """
    Return facts whose topic or fact text overlaps with the query.
    Uses simple Jaccard similarity on word tokens.
    Returns a list of (score, fact) tuples sorted by relevance.
    """
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    scored = []
    for fact in facts:
        # Compare query against both topic and fact text
        topic_tokens = _tokenise(fact.get("topic", ""))
        fact_tokens  = _tokenise(fact.get("fact",  ""))
        combined     = topic_tokens | fact_tokens

        if not combined:
            continue

        # Jaccard similarity: intersection / union
        intersection = len(query_tokens & combined)
        union        = len(query_tokens | combined)
        score        = intersection / union if union else 0.0

        if score >= threshold:
            scored.append((score, fact))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max_results]


def format_facts_for_prompt(facts: list, n: int = FACTS_TO_INJECT) -> str:
    """
    Format the most recent N facts into a string to inject into context.
    The agent sees this at the start of each session so it can draw on
    its knowledge base when answering questions.
    """
    if not facts:
        return ""

    recent = facts[-n:]
    lines = [f"## Knowledge base ({len(facts)} total facts):\n"]
    for f in recent:
        lines.append(f"[Fact #{f['id']} | {f['topic']}]")
        lines.append(f"  {f['fact']}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# DISCUSSION ENGINE
# Handles the interactive parent-child discussion flow.
# The agent presents a finding, you discuss it, and the agreed
# answer is committed as a fact.
# ─────────────────────────────────────────────

def run_discussion(topic: str, finding: str, model_fn, facts: list,
                   source: str = "explored") -> bool:
    """
    Run an interactive discussion between the agent and user about a finding.

    Arguments:
        topic      — short label for what's being discussed
        finding    — what the agent found or believes
        model_fn   — callable that takes a prompt string and returns a response
                     string (used to generate the agent's discussion replies)
        facts      — the live facts list to append to if confirmed
        source     — where this finding came from

    Returns True if a fact was saved, False if the user declined.
    """
    print("\n" + "─" * 60)
    print(f"🔍 \033[1mAgent finding on: {topic}\033[0m")
    print(f"\n{finding}")
    print("\n" + "─" * 60)

    # Check if we already have a fact on this topic
    existing = find_relevant_facts(topic, facts, threshold=0.5)
    if existing:
        print(f"\n📚 Related fact(s) already in knowledge base:")
        for score, f in existing:
            print(f"   [#{f['id']}] {f['topic']}: {f['fact'][:120]}...")
        print()

    print("💬 Let's discuss this. You can:")
    print("   • Correct the agent   • Add context   • Confirm it's right")
    print("   • Type 'skip' to discard   • Type 'save' to save as-is\n")

    # Build a conversation history for the discussion
    discussion_history = [
        {"role": "system", "content": (
            "You are in a Socratic learning discussion with your user. "
            "You have just researched a topic and presented a finding. "
            "Now discuss it with the user, ask clarifying questions, "
            "accept corrections gracefully, and work toward a clear agreed fact. "
            "Keep responses concise — 2-4 sentences. "
            "When the user seems satisfied, summarise the agreed fact in one clear sentence "
            "prefixed with 'AGREED FACT: '."
        )},
        {"role": "assistant", "content": (
            f"I've been researching '{topic}' and here's what I found:\n\n"
            f"{finding}\n\n"
            f"What do you think? Is this correct, or would you like to clarify anything?"
        )},
    ]

    agreed_fact = None
    turn = 0
    max_turns = 10  # prevent infinite loops

    while turn < max_turns:
        try:
            user_reply = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Discussion interrupted]")
            return False

        if not user_reply:
            continue

        # User commands
        if user_reply.lower() == "skip":
            print("\n[Knowledge] Discarded — no fact saved.\n")
            return False

        if user_reply.lower() == "save":
            # Save the original finding as-is
            agreed_fact = finding
            break

        # Add user reply to history and get agent response
        discussion_history.append({"role": "user", "content": user_reply})

        try:
            agent_reply = model_fn(discussion_history)
        except Exception as e:
            print(f"[Knowledge] Model error during discussion: {e}")
            agent_reply = "I'm having trouble responding. Could you summarise what we've agreed on?"

        print(f"\n🤖 Agent: {agent_reply}\n")
        discussion_history.append({"role": "assistant", "content": agent_reply})

        # Check if the agent produced an agreed fact
        match = re.search(r"AGREED FACT:\s*(.+?)(?:\n|$)", agent_reply, re.IGNORECASE)
        if match:
            agreed_fact = match.group(1).strip()
            print(f"\n✨ Proposed fact: \"{agreed_fact}\"")
            confirm = input("Save this fact? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                break
            else:
                agreed_fact = None
                print("Ok, let's keep discussing.\n")

        turn += 1

    if agreed_fact is None:
        # Ask user to summarise if we ran out of turns without agreement
        print("\n[Knowledge] Let's wrap up. How would you summarise what we agreed on?")
        try:
            agreed_fact = input("Fact: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Knowledge] Discarded.\n")
            return False

        if not agreed_fact:
            print("[Knowledge] No fact saved.\n")
            return False

    # Save the agreed fact
    fact = add_fact(facts, topic, finding, agreed_fact, source=source)
    print(f"\n✅ \033[32mFact #{fact['id']} saved to knowledge base.\033[0m")
    print(f"   Topic : {fact['topic']}")
    print(f"   Fact  : {fact['fact']}\n")
    return True


# ─────────────────────────────────────────────
# EXPLORATION ENGINE
# Handles the "explore: <topic>" command and the automatic gap-detection
# that fires when the agent encounters something it doesn't have a fact for.
# ─────────────────────────────────────────────

def explore_topic(topic: str, model_fn, search_fn, facts: list) -> bool:
    """
    Agent autonomously researches a topic using web search, forms a finding,
    then enters discussion with the user before committing to facts.

    Arguments:
        topic     — the topic to explore
        model_fn  — callable(messages) -> str for agent replies
        search_fn — callable(query) -> str for web search results
        facts     — live facts list

    Returns True if a fact was saved.
    """
    print(f"\n🌐 \033[1mExploring: {topic}\033[0m\n")

    # Step 1: web search
    print("   Searching the web...")
    try:
        search_results = search_fn(topic)
    except Exception as e:
        search_results = f"Search failed: {e}"
        print(f"   Search error: {e}")

    # Step 2: agent synthesises a finding from search results
    synthesis_prompt = [
        {"role": "system", "content": (
            "You are a research assistant. Given web search results about a topic, "
            "write a concise factual finding in 2-4 sentences. "
            "Be precise. Do not pad with caveats. "
            "Start with the most important fact."
        )},
        {"role": "user", "content": (
            f"Topic: {topic}\n\n"
            f"Search results:\n{search_results}\n\n"
            f"Write a concise factual finding about this topic."
        )},
    ]

    print("   Synthesising finding...")
    try:
        finding = model_fn(synthesis_prompt)
    except Exception as e:
        print(f"[Knowledge] Could not synthesise finding: {e}")
        return False

    # Step 3: run discussion
    return run_discussion(topic, finding, model_fn, facts, source="explored")


def check_and_flag_gap(query: str, response: str, model_fn, facts: list) -> bool:
    """
    Called automatically after the agent answers a question.
    Asks the model whether the response revealed a knowledge gap
    that should be explored and saved as a fact.

    Returns True if a fact was saved, False otherwise.
    """
    # Don't flag for very short responses or simple tasks
    if len(response) < 100:
        return False

    # Ask the model to self-assess
    gap_prompt = [
        {"role": "system", "content": (
            "You are reviewing an agent's response to decide if it contains "
            "a significant factual claim that should be saved to a knowledge base. "
            "Reply with only: YES: <topic> or NO"
        )},
        {"role": "user", "content": (
            f"User asked: {query}\n\n"
            f"Agent responded: {response[:500]}\n\n"
            "Does this response contain a significant factual claim worth saving? "
            "If YES, give a short topic label (3-6 words). Reply YES: <topic> or NO."
        )},
    ]

    try:
        gap_check = model_fn(gap_prompt)
    except Exception:
        return False

    match = re.match(r"YES:\s*(.+)", gap_check.strip(), re.IGNORECASE)
    if not match:
        return False

    suggested_topic = match.group(1).strip()

    # Check if we already have this fact
    existing = find_relevant_facts(suggested_topic, facts, threshold=0.5)
    if existing:
        return False  # already know this

    # Ask user if they want to discuss and save it
    print(f"\n💡 \033[33mI noticed a fact worth saving about: \"{suggested_topic}\"\033[0m")
    try:
        choice = input("   Would you like to discuss and save it? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if choice not in ("y", "yes"):
        return False

    # Run discussion with the response as the finding
    return run_discussion(suggested_topic, response[:800], model_fn, facts,
                          source="automatic")


# ─────────────────────────────────────────────
# KNOWLEDGE MANAGEMENT COMMANDS
# Utility functions for listing, viewing, editing, and deleting facts.
# These are triggered by typing commands in the agent REPL.
# ─────────────────────────────────────────────

def handle_knowledge_command(command: str, facts: list) -> bool:
    """
    Handle knowledge management commands typed directly in the REPL.
    Returns True if the command was handled (so the main loop can skip
    sending it to the agent).

    Commands:
        facts list            — show all facts
        facts show <id>       — show full detail of a fact
        facts delete <id>     — delete a fact
        facts search <query>  — search facts
        facts count           — show total count
    """
    cmd = command.strip().lower()

    if cmd == "facts list":
        if not facts:
            print("\n📚 Knowledge base is empty.\n")
        else:
            print(f"\n📚 Knowledge base — {len(facts)} fact(s):\n")
            for f in facts:
                print(f"  #{f['id']:3d} [{f['date']}] {f['topic']}")
                print(f"       {f['fact'][:100]}{'...' if len(f['fact']) > 100 else ''}")
                print()
        return True

    if cmd == "facts count":
        print(f"\n📚 {len(facts)} fact(s) in knowledge base.\n")
        return True

    if cmd.startswith("facts show "):
        try:
            fid = int(cmd.split()[-1])
            matches = [f for f in facts if f["id"] == fid]
            if matches:
                f = matches[0]
                print(f"\n📄 Fact #{f['id']}")
                print(f"   Topic   : {f['topic']}")
                print(f"   Date    : {f['date']}")
                print(f"   Source  : {f['source']}")
                print(f"   Finding : {f['finding']}")
                print(f"   Fact    : {f['fact']}\n")
            else:
                print(f"\n[Knowledge] No fact with ID {fid}.\n")
        except ValueError:
            print("\n[Knowledge] Usage: facts show <id>\n")
        return True

    if cmd.startswith("facts delete "):
        try:
            fid = int(cmd.split()[-1])
            if delete_fact(facts, fid):
                print(f"\n🗑️  Fact #{fid} deleted.\n")
            else:
                print(f"\n[Knowledge] No fact with ID {fid}.\n")
        except ValueError:
            print("\n[Knowledge] Usage: facts delete <id>\n")
        return True

    if cmd.startswith("facts search "):
        query = command[len("facts search "):].strip()
        results = find_relevant_facts(query, facts, threshold=0.1, max_results=10)
        if results:
            print(f"\n🔎 Facts matching '{query}':\n")
            for score, f in results:
                print(f"  #{f['id']:3d} [{score:.2f}] {f['topic']}")
                print(f"       {f['fact'][:120]}{'...' if len(f['fact']) > 120 else ''}")
                print()
        else:
            print(f"\n[Knowledge] No facts found matching '{query}'.\n")
        return True

    return False  # not a knowledge command


# ─────────────────────────────────────────────
# FACT INJECTION
# Builds the facts block to prepend to agent prompts so the agent
# can draw on its knowledge base when answering questions.
# ─────────────────────────────────────────────

def build_facts_prompt_block(facts: list) -> str:
    """
    Returns a formatted string of all facts to inject into the agent's
    context. Called at startup and included in the memory prefix.
    """
    return format_facts_for_prompt(facts, FACTS_TO_INJECT)

#!/usr/bin/env python3
"""
Local Agentic Note-Taking System

Captures your thoughts and notes, then uses Claude to distill them
into themes, insights, and a clear path forward.

Usage:
  python notes_agent.py add "your thought here"
  python notes_agent.py show
  python notes_agent.py distill
  python notes_agent.py chat
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import anthropic

# Notes are stored in ~/.notes_agent/notes.json
NOTES_FILE = Path.home() / ".notes_agent" / "notes.json"
MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """You are a personal thought distillation assistant. Your role is to:

1. **Capture** — Help users quickly record thoughts, ideas, and observations
2. **Organize** — Categorize and tag notes to surface patterns
3. **Distill** — Synthesize accumulated notes into clear insights and themes
4. **Pathfind** — Identify actionable next steps and a clear path forward

When asked to distill or analyze notes:
- Look for recurring themes and patterns across all notes
- Identify open questions, tensions, or decisions that need making
- Surface the most important insights
- Suggest concrete next actions grouped by theme
- Present findings in a clear, structured format with headers

Be concise but deeply insightful. Transform scattered thoughts into clarity."""


# ── Storage helpers ────────────────────────────────────────────────────────

def _ensure_notes_file() -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES_FILE.exists():
        NOTES_FILE.write_text(json.dumps({"notes": []}, indent=2))


def _load_notes() -> dict:
    _ensure_notes_file()
    return json.loads(NOTES_FILE.read_text())


def _save_notes(data: dict) -> None:
    NOTES_FILE.write_text(json.dumps(data, indent=2))


# ── Tool implementations ───────────────────────────────────────────────────

def _add_note(content: str, tags: list[str] | None = None) -> dict:
    data = _load_notes()
    note = {
        "id": len(data["notes"]) + 1,
        "content": content,
        "tags": tags or [],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    data["notes"].append(note)
    _save_notes(data)
    return {"success": True, "note_id": note["id"], "message": f"Note #{note['id']} saved."}


def _read_all_notes() -> dict:
    data = _load_notes()
    if not data["notes"]:
        return {"notes": [], "count": 0, "message": "No notes saved yet."}
    return {"notes": data["notes"], "count": len(data["notes"])}


def _search_notes(query: str) -> dict:
    data = _load_notes()
    q = query.lower()
    matches = [
        n for n in data["notes"]
        if q in n["content"].lower()
        or any(q in tag.lower() for tag in n.get("tags", []))
    ]
    return {"matches": matches, "count": len(matches), "query": query}


def _get_stats() -> dict:
    data = _load_notes()
    notes = data["notes"]
    if not notes:
        return {"total": 0, "message": "No notes saved yet."}
    dates = [n["timestamp"] for n in notes]
    all_tags = list({tag for n in notes for tag in n.get("tags", [])})
    return {
        "total": len(notes),
        "oldest": min(dates),
        "newest": max(dates),
        "all_tags": all_tags,
    }


def _execute_tool(name: str, inputs: dict) -> str:
    if name == "add_note":
        result = _add_note(inputs["content"], inputs.get("tags"))
    elif name == "read_all_notes":
        result = _read_all_notes()
    elif name == "search_notes":
        result = _search_notes(inputs["query"])
    elif name == "get_stats":
        result = _get_stats()
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ── Tool schema ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "add_note",
        "description": "Save a thought, idea, or observation with optional tags.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The thought or note to capture",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization (e.g. ['work', 'idea'])",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_all_notes",
        "description": "Retrieve all saved notes and thoughts.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_notes",
        "description": "Search notes by keyword or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get statistics about the notes collection (count, date range, tags).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ── Agentic loop ───────────────────────────────────────────────────────────

def _run_agent(user_message: str, verbose: bool = False) -> str:
    """Run one turn of the agentic loop and return Claude's final text response."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
        )

        if verbose:
            for block in response.content:
                if block.type == "thinking":
                    preview = block.thinking[:300].replace("\n", " ")
                    print(f"\n[thinking] {preview}…", file=sys.stderr)

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")

        if response.stop_reason != "tool_use":
            break

        # Execute tool calls
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_use_blocks:
            if verbose:
                print(f"[tool] {tu.name}({json.dumps(tu.input)})", file=sys.stderr)
            result = _execute_tool(tu.name, tu.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})

    return ""


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> None:
    text = " ".join(args.note)
    result = _add_note(text)
    print(f"✓ {result['message']}")


def cmd_show(args: argparse.Namespace) -> None:
    data = _load_notes()
    notes = data["notes"]
    if not notes:
        print("No notes yet.  Add one:  python notes_agent.py add <thought>")
        return

    print(f"\n{'─' * 50}")
    print(f"  {len(notes)} notes  —  {NOTES_FILE}")
    print(f"{'─' * 50}\n")

    for note in notes[-30:]:
        ts = note["timestamp"][:16].replace("T", " ")
        tags = f"  [{', '.join(note['tags'])}]" if note.get("tags") else ""
        print(f"[{note['id']}] {ts}{tags}")
        print(f"    {note['content']}")
        print()


def cmd_distill(args: argparse.Namespace) -> None:
    data = _load_notes()
    if not data["notes"]:
        print("No notes to distill yet.  Start capturing thoughts first.")
        return

    print(f"Analyzing {len(data['notes'])} notes…\n")
    result = _run_agent(
        "Please read all my notes, then distill them into key themes, "
        "important insights, open questions, and a clear path forward "
        "with concrete next actions.",
        verbose=args.verbose,
    )
    print(result)


def cmd_chat(args: argparse.Namespace) -> None:
    print("Chat mode  —  type 'exit' to quit.\n")
    client = anthropic.Anthropic()
    messages: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Inner agentic loop for this turn
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
            )

            if response.stop_reason == "end_turn":
                text = next((b.text for b in response.content if b.type == "text"), "")
                print(f"\nAssistant: {text}\n")
                # Store only text in history to keep context clean
                messages.append({"role": "assistant", "content": text})
                break

            if response.stop_reason != "tool_use":
                break

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_use_blocks:
                result = _execute_tool(tu.name, tu.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                )

            messages.append({"role": "user", "content": tool_results})


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Agentic Note-Taking System — capture thoughts, distill clarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python notes_agent.py add "need to revisit the pricing model"
  python notes_agent.py add "idea: weekly async standups instead of daily"
  python notes_agent.py show
  python notes_agent.py distill
  python notes_agent.py chat
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Quickly capture a thought or note")
    p_add.add_argument("note", nargs="+", help="The thought to capture (no quotes needed)")
    p_add.set_defaults(func=cmd_add)

    # show
    p_show = sub.add_parser("show", help="Show your recent notes")
    p_show.set_defaults(func=cmd_show)

    # distill
    p_distill = sub.add_parser("distill", help="Distill notes into insights and a path forward")
    p_distill.add_argument("--verbose", "-v", action="store_true", help="Show tool calls")
    p_distill.set_defaults(func=cmd_distill)

    # chat
    p_chat = sub.add_parser("chat", help="Chat interactively with the agent about your notes")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

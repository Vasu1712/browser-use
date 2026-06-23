#!/usr/bin/env python3
"""
Converts a browser-use action cache (JSONL) into a deterministic
optexity automation JSON — no LLM needed on replay.
Locator priority (mirrors optexity's own scoring)
"""

import json
import re
import sys

# Action types that are non-interactive noise — always filter out
SKIP_ACTIONS = {"done", "scroll", "wait", "go_back", "extract_data"}


def classify_verdict(eval_text: str) -> str:
    """Classify an eval_previous_goal string as 'success' / 'failure' / 'unknown'.

    Mirrors the agent's own logic (success checked before failure).
    """
    if not eval_text:
        return "unknown"
    low = eval_text.lower()
    if "success" in low:
        return "success"
    if "failure" in low:
        return "failure"
    return "unknown"


def compute_dead_end_steps(actions: list) -> set:
    """Identify steps whose actions led to a dead end.

    eval_previous_goal at step K evaluates the previous ACTED step's actions. So if the
    next acted step's verdict is 'failure', the current step's actions were a dead end
    (e.g. clicked a premium image, hit a paywall, had to recover). Those should be dropped.

    Backward compatible: if no records carry eval_previous_goal (older caches), no step
    has a known verdict, so nothing is flagged and filtering behaves exactly as before.
    """
    step_verdict = {}
    for a in actions:
        s = a.get("step_number")
        if s is not None and s not in step_verdict:
            step_verdict[s] = classify_verdict(a.get("eval_previous_goal", ""))

    sorted_steps = sorted(step_verdict)
    dead_end = set()
    for i, s in enumerate(sorted_steps[:-1]):
        next_s = sorted_steps[i + 1]
        if step_verdict.get(next_s) == "failure":
            dead_end.add(s)
    return dead_end


def is_stable_id(el_id: str) -> bool:
    """Check if an element ID looks stable vs dynamically generated."""
    if not el_id:
        return False
    unstable_patterns = [
        r"^base-ui-",
        r"^:r",
        r"^react-",
        r"^radix-",
        r"^mui-",
        r"^__next",
        r"_r_[a-z0-9]+_$",
    ]
    for pattern in unstable_patterns:
        if re.search(pattern, el_id):
            return False
    return True


def clean_visible_text(text: str) -> str:
    """Extract meaningful text from visible_text, removing icon descriptions."""
    if not text:
        return ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    meaningful = [l for l in lines if not re.match(r"^An?\s+\w+$", l)]
    return meaningful[0] if meaningful else (lines[-1] if lines else "")


def build_locator(attrs: dict) -> str:
    """Synthesize the best stable Playwright locator from cached element attributes."""
    tag = attrs.get("tag", "input")
    el_id = attrs.get("id", "")
    name = attrs.get("name", "")
    aria = attrs.get("aria_label", "")
    placeholder = attrs.get("placeholder", "")
    role = attrs.get("role", "")
    raw_text = attrs.get("visible_text", "")
    el_type = attrs.get("type", "")

    text = clean_visible_text(raw_text)

    if is_stable_id(el_id):
        return f'locator("#{el_id}").first'
    if name:
        return f"locator(\"{tag}[name='{name}']\").first"
    if aria:
        return f'get_by_label("{aria}").first'
    if placeholder:
        return f'get_by_placeholder("{placeholder}").first'
    if role and text:
        return f'get_by_role("{role}", name="{text}").first'
    if el_type:
        return f"locator(\"{tag}[type='{el_type}']\").first"
    return f'locator("{tag}").first'


def build_prompt_instructions(action: dict) -> str:
    """Human-readable description of what this node does."""
    attrs = action.get("element_attributes", {})
    name = attrs.get("name", "")
    raw_text = attrs.get("visible_text", "")
    text = clean_visible_text(raw_text)

    if action["action_type"] == "input_text":
        field_desc = name or text or "the field"
        return f"Enter '{action.get('text', '')}' into {field_desc}"
    elif action["action_type"] in ("click", "click_element"):
        target_desc = text or name or "the element"
        return f"Click on {target_desc}"
    return ""


def build_node(action: dict) -> dict | None:
    """Convert one cached action into an optexity automation node."""
    attrs = action.get("element_attributes", {})
    locator = build_locator(attrs)

    if action["action_type"] == "input_text":
        return {
            "type": "action_node",
            "interaction_action": {
                "input_text": {
                    "command": locator,
                    "prompt_instructions": build_prompt_instructions(action),
                    "input_text": action.get("text", ""),
                }
            },
        }
    elif action["action_type"] in ("click", "click_element"):
        return {
            "type": "action_node",
            "interaction_action": {
                "click_element": {
                    "command": locator,
                    "prompt_instructions": build_prompt_instructions(action),
                }
            },
        }
    return None


def effective_actions(actions: list) -> list:
    """Filter a raw cache action list down to the effective, replayable actions.

    Drops non-interactive noise (SKIP_ACTIONS), dead-end steps (disconfirmed by the next
    step's Failure verdict), cache-flagged duplicates, and same-target repeats. Reused by
    the iterative optimizer to (re)synthesize deterministic nodes from a node's cache.
    """
    dead_end_steps = compute_dead_end_steps(actions)
    effective = []
    seen = set()
    for a in actions:
        atype = a["action_type"]
        if atype in SKIP_ACTIONS:
            continue
        if a.get("step_number") in dead_end_steps:
            continue
        if a.get("is_duplicate"):
            continue
        attrs = a.get("element_attributes", {})
        el_key = attrs.get("name", "") or attrs.get("visible_text", "") or str(a.get("index", ""))
        dedup_key = (atype, el_key, a.get("text", ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        effective.append(a)
    return effective


def convert(cache_path: str, output_path: str, source_path: str = None):
    actions = []
    with open(cache_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                actions.append(json.loads(line))

    # --- Load parameters from source automation if provided ---
    parameters = {"input_parameters": {}, "generated_parameters": {}}
    if source_path:
        with open(source_path, "r") as f:
            source = json.load(f)
        parameters = source.get("parameters", parameters)
        print(f"Source: {source_path}")
        print(f"  Carried over parameters: {json.dumps(parameters)}")

    # --- Identify dead-end steps (actions disconfirmed by the next step's Failure verdict) ---
    dead_end_steps = compute_dead_end_steps(actions)

    # --- Filter ---
    effective = []
    seen = set()
    skipped_types = []
    for a in actions:
        atype = a["action_type"]

        if atype in SKIP_ACTIONS:
            skipped_types.append(atype)
            continue

        if a.get("step_number") in dead_end_steps:
            skipped_types.append(f"{atype}(deadend)")
            continue

        if a.get("is_duplicate"):
            skipped_types.append(f"{atype}(dup)")
            continue

        attrs = a.get("element_attributes", {})
        el_key = attrs.get("name", "") or attrs.get("visible_text", "") or str(a.get("index", ""))
        dedup_key = (atype, el_key, a.get("text", ""))
        if dedup_key in seen:
            skipped_types.append(f"{atype}(dedup)")
            continue
        seen.add(dedup_key)
        effective.append(a)

    # --- Get URL from first action ---
    url = effective[0]["url"] if effective else ""

    # --- Build automation ---
    nodes = []
    for a in effective:
        node = build_node(a)
        if node:
            nodes.append(node)

    automation = {
        "url": url,
        "parameters": parameters,
        "nodes": nodes,
    }

    with open(output_path, "w") as f:
        json.dump(automation, f, indent=2)

    # --- Report ---
    print(f"\nCache: {len(actions)} total actions")
    print(f"  Skipped: {len(skipped_types)} ({', '.join(skipped_types) if skipped_types else 'none'})")
    print(f"  Effective: {len(effective)} -> {len(nodes)} nodes")
    print(f"Output: {output_path}")
    print()
    for i, n in enumerate(nodes):
        ia = n["interaction_action"]
        action_type = list(ia.keys())[0]
        details = ia[action_type]
        cmd = details["command"]
        desc = details["prompt_instructions"]
        print(f"  Node {i}: [{action_type}] {cmd}")
        print(f"           {desc}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cache_to_automation.py <cache.jsonl> [output.json] [--source original.json]")
        sys.exit(1)

    cache_file = sys.argv[1]
    output_file = "test_automation_cached.json"
    source_file = None

    # Parse remaining args
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source_file = args[i + 1]
            i += 2
        else:
            output_file = args[i]
            i += 1

    convert(cache_file, output_file, source_file)
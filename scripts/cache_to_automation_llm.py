#!/usr/bin/env python3
"""
LLM-based cache -> optexity automation builder, grounded in the real optexity schema
and validated by Pydantic (the SAME validation the optexity worker runs).

This is an alternative to the rule-based scripts/cache_to_automation.py. Instead of
hand-coded filtering + locator synthesis, it hands the cached action log to an LLM and
asks it to emit a valid optexity Automation JSON, then validates the result with
`Automation.model_validate_json()` in a retry loop. The LLM runs only here, at build
time — the produced automation replays deterministically with NO LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Real optexity schema — the ground truth for both the prompt and validation.
from optexity.schema.automation import Automation
from pydantic import ValidationError

# Reuse browser-use's existing LLM abstraction (importing browser_use also loads .env).
from browser_use.llm import AssistantMessage, ChatGoogle, SystemMessage, UserMessage

DEFAULT_MODEL = 'gemini-flash-latest'
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0


def load_actions(cache_path: str) -> list[dict]:
	"""Read the JSONL action cache into a list of records."""
	actions: list[dict] = []
	with open(cache_path, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line:
				actions.append(json.loads(line))
	return actions


def load_parameters(source_path: str | None) -> dict:
	"""Carry over the parameters block from a source automation (mirrors the rule-based converter)."""
	parameters = {'input_parameters': {}, 'generated_parameters': {}}
	if source_path:
		with open(source_path, encoding='utf-8') as f:
			source = json.load(f)
		parameters = source.get('parameters', parameters)
		print(f'Source: {source_path}')
		print(f'  Carried over parameters: {json.dumps(parameters)}')
	return parameters


def build_system_prompt(schema_json: str, parameters: dict) -> str:
	"""System prompt grounded in the real Automation JSON schema."""
	param_keys = list((parameters or {}).get('input_parameters', {}).keys())
	param_hint = (
		f'The automation declares these input parameters: {param_keys}. Where a typed value '
		f'corresponds to one of them, emit the placeholder "{{name[0]}}" (e.g. "{{search_query[0]}}") '
		f'as the input_text instead of the literal value, so the automation stays parameterized.'
		if param_keys
		else 'There are no input parameters; use literal values from the action log.'
	)
	return f"""You convert a recorded log of browser actions (produced by an LLM-driven agent) into a \
DETERMINISTIC optexity automation JSON that can be replayed without any LLM.

The output MUST conform exactly to this JSON schema (the optexity `Automation` model):

{schema_json}

HOW TO BUILD THE AUTOMATION:
- Produce a top-level object with at least "url", "parameters", and "nodes".
- "url" is the starting URL (take it from the first effective action's "url").
- Each kept action becomes a node: {{"type": "action_node", "interaction_action": {{ ... }}}}.
  - A text entry -> "input_text": {{"command": <locator>, "prompt_instructions": <short human description>, "input_text": <value>}}
  - A click     -> "click_element": {{"command": <locator>, "prompt_instructions": <short human description>}}

LOCATOR RULES (critical):
- "command" is executed as `page.<command>` (Python Playwright). It MUST be a valid Playwright
  Python locator expression ending in `.first`, e.g.
  `locator("[name='searchKeyword']").first` or `get_by_role("button", name="Search").first`.
  NEVER output a bare CSS selector or a raw tag name.
- Prefer the MOST STABLE locator available, in this priority order:
  1. id (only if stable)      -> locator("#id").first
  2. name                     -> locator("tag[name='value']").first
  3. aria-label               -> get_by_label("value").first
  4. placeholder              -> get_by_placeholder("value").first
  5. role + visible text      -> get_by_role("role", name="text").first
- AVOID dynamically-generated ids; treat these as unstable and fall back to the next option:
  anything matching base-ui-*, :r*, react-*, radix-*, mui-*, __next*, or ids like `_r_xx_`.
- AVOID hardcoding dynamic data (like article titles, dates, or specific IDs) into the locator text. 
  If the action was clicking the "first item in a list", prefer using CSS selectors with an index or generic roles (e.g., locator(".search-result-title").first or get_by_role("listitem").first) rather than exact text matches.

WHICH ACTIONS TO KEEP (filter aggressively — replay must be the minimal happy path):
- DROP any action with "is_duplicate": true.
- DROP non-interactive noise: done, scroll, wait, go_back, extract_data, navigate-for-recovery.
- Use "eval_previous_goal": it evaluates the PREVIOUS acted step. If a later step's
  eval_previous_goal contains a Failure verdict, the action(s) it is judging were a DEAD END
  (e.g. clicked a paywalled item and had to back out) — DROP those dead-end actions.
- "next_goal" and "memory" are context to help you understand intent; they are not emitted.
- Keep only the actions that form the shortest successful path to the task's goal.

PARAMETERS:
- Copy the "parameters" block as given (it will also be re-applied after validation): {json.dumps(parameters)}.
- {param_hint}

OUTPUT FORMAT:
- Output ONLY the raw JSON object. No prose, no explanation, no markdown code fences."""


def build_user_message(actions: list[dict]) -> str:
	"""The user message: the raw cache actions for the model to convert."""
	return (
		'Here is the recorded action log (one JSON object per executed action, in order). '
		'Convert it into a valid optexity automation JSON following the system instructions:\n\n'
		f'{json.dumps(actions, indent=2)}'
	)


def extract_json(text: str) -> str:
	"""Defensively strip markdown fences / prose and return the JSON object substring."""
	t = text.strip()
	if t.startswith('```'):
		# remove the opening fence line (``` or ```json) and any trailing fence
		t = t.split('\n', 1)[1] if '\n' in t else t
		if t.rstrip().endswith('```'):
			t = t.rstrip()[:-3]
	t = t.strip()
	# Fall back to the outermost braces if there is still surrounding text.
	start, end = t.find('{'), t.rfind('}')
	if start != -1 and end != -1 and end > start:
		return t[start : end + 1]
	return t


async def build(
	cache_path: str,
	source_path: str | None,
	model: str,
	max_retries: int,
	force_literal_inputs: bool = True,
) -> dict | None:
	"""Run the LLM build+validation loop and return the validated automation dict (or None)."""
	actions = load_actions(cache_path)
	parameters = load_parameters(source_path)
	schema_json = json.dumps(Automation.model_json_schema())

	llm = ChatGoogle(model=model, temperature=DEFAULT_TEMPERATURE)
	messages: list = [
		SystemMessage(content=build_system_prompt(schema_json, parameters)),
		UserMessage(content=build_user_message(actions)),
	]

	print(f'\nCache: {len(actions)} actions -> building automation with {model} (temp={DEFAULT_TEMPERATURE})')

	last_error = None
	for attempt in range(1, max_retries + 1):
		result = await llm.ainvoke(messages)
		raw = result.completion if isinstance(result.completion, str) else str(result.completion)
		candidate = extract_json(raw)

		try:
			# The exact call the optexity worker uses to load an automation.
			Automation.model_validate_json(candidate)
		except ValidationError as e:
			last_error = e
			print(f'  Attempt {attempt}/{max_retries}: validation FAILED ({len(e.errors())} error(s)); asking for a fix...')
			messages.append(AssistantMessage(content=raw))
			messages.append(
				UserMessage(
					content=(
						'That JSON failed `Automation.model_validate_json()` with the following '
						f'pydantic errors:\n\n{e}\n\nReturn ONLY corrected raw JSON that fixes every '
						'error. No prose, no markdown fences.'
					)
				)
			)
			continue

		# Validated. Deterministically re-apply the source parameters, then re-validate to be safe.
		obj = json.loads(candidate)
		if source_path:
			obj['parameters'] = parameters

		# Force each input_text node to the literal value from the cache, overriding any
		# {param[0]} placeholder the model emitted. Matched to input_text nodes in order.
		if force_literal_inputs:
			cache_inputs = [
				a['text']
				for a in actions
				if a.get('action_type') == 'input_text' and not a.get('is_duplicate') and a.get('text') is not None
			]
			i = 0
			for node in obj.get('nodes', []):
				ia = node.get('interaction_action', {})
				if isinstance(ia, dict) and 'input_text' in ia and i < len(cache_inputs):
					ia['input_text']['input_text'] = cache_inputs[i]
					i += 1

		Automation.model_validate(obj)  # ensure still valid after the overrides

		nodes = obj.get('nodes', [])
		print(f'  Attempt {attempt}/{max_retries}: validation OK -> {len(nodes)} nodes')
		return obj

	print(f'\nFAILED: no valid automation after {max_retries} attempts.', file=sys.stderr)
	if last_error is not None:
		print(str(last_error), file=sys.stderr)
	return None


def main() -> None:
	parser = argparse.ArgumentParser(description='LLM-based cache -> optexity automation builder (Pydantic-validated).')
	parser.add_argument('cache', help='Path to the action cache JSONL file')
	parser.add_argument('output', nargs='?', default=None, help='Output automation JSON path')
	parser.add_argument('--source', default=None, help='Source automation JSON to carry the parameters block from')
	parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Gemini model (default: {DEFAULT_MODEL})')
	parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help='Validation retries (default: 3)')
	parser.add_argument(
		'--literal-inputs',
		action='store_true',
		help="Bake literal cached values into input_text nodes instead of {param[0]} placeholders",
	)
	args = parser.parse_args()

	output = args.output or f'{Path(args.cache).stem}_llm.json'
	automation = asyncio.run(build(args.cache, args.source, args.model, args.max_retries, args.literal_inputs))
	if automation is None:
		sys.exit(1)
	Path(output).write_text(json.dumps(automation, indent=2), encoding='utf-8')
	print(f'Output: {output}')


if __name__ == '__main__':
	main()

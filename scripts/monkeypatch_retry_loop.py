#!/usr/bin/env python3
"""
Offline, deterministic verification of the Pydantic retry loop in cache_to_automation_llm.py.

Why this exists: trying to trigger the retry loop by altering the system prompt does not
work — with the schema in-context and temperature=0, the model returns valid JSON on the
first attempt, so the loop never reaches its `except ValidationError` branch. To exercise
that branch we must GUARANTEE a ValidationError, independent of any model.

This script monkeypatches the module-level `ChatGoogle` in cache_to_automation_llm with a
fake whose `ainvoke` returns a scripted sequence of completions (invalid -> ... -> valid).
The real ChatGoogle is never constructed, so there is no network call and no GOOGLE_API_KEY
is needed. The production script is not modified.

Run with optexity's venv python (the imported module needs both packages):

    /Users/vasu/Desktop/Projects/optexity/.venv/bin/python scripts/verify_retry_loop.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

# Make the script-under-test importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_to_automation_llm as m  # noqa: E402

# --- Scripted completions -------------------------------------------------------------

# A minimal automation that passes Automation.model_validate_json().
VALID = json.dumps(
	{
		'url': 'https://example.com/',
		'parameters': {'input_parameters': {}, 'generated_parameters': {}},
		'nodes': [
			{
				'type': 'action_node',
				'interaction_action': {
					'input_text': {
						'command': 'locator("#q").first',
						'prompt_instructions': 'enter query',
						'input_text': 'AI',
					}
				},
			}
		],
	}
)

# Well-formed node dict but the interaction_action is empty -> the InteractionAction
# "exactly one ... must be provided" rule raises ValidationError. (Survives optexity's
# migrate_old_nodes pre-validator, which only chokes on non-dict node items.)
INVALID_EMPTY_NODE = json.dumps(
	{
		'url': 'x',
		'parameters': {'input_parameters': {}, 'generated_parameters': {}},
		'nodes': [{'type': 'action_node', 'interaction_action': {}}],
	}
)
# Not JSON at all -> model_validate_json also raises ValidationError.
INVALID_JUNK = 'this is not json at all'


class FakeChat:
	"""Stand-in for ChatGoogle that returns scripted completions and records each call."""

	def __init__(self, scripted: list[str]):
		self.scripted = scripted
		self.calls = 0
		self.message_counts: list[int] = []
		self.last_texts: list[str] = []

	async def ainvoke(self, messages, *args, **kwargs):
		self.message_counts.append(len(messages))
		last = messages[-1]
		self.last_texts.append(getattr(last, 'text', str(last)))
		# Clamp so an over-eager loop still gets a defined (last) response.
		text = self.scripted[min(self.calls, len(self.scripted) - 1)]
		self.calls += 1
		return SimpleNamespace(completion=text)


def _write_temp_cache() -> str:
	"""A 1-line JSONL cache; content is irrelevant — the fake LLM ignores the actions."""
	fd, path = tempfile.mkstemp(suffix='.jsonl', prefix='verify_retry_')
	with os.fdopen(fd, 'w', encoding='utf-8') as f:
		f.write(json.dumps({'step_number': 1, 'action_type': 'input_text', 'index': 1, 'text': 'AI'}) + '\n')
	return path


def _run(scripted: list[str], max_retries: int) -> tuple[FakeChat, object]:
	fake = FakeChat(scripted)
	m.ChatGoogle = lambda *a, **k: fake  # inject before build() constructs the client
	result = asyncio.run(m.build(CACHE_PATH, None, 'fake-model', max_retries))
	return fake, result


CACHE_PATH = _write_temp_cache()
_failures: list[str] = []


def check(label: str, ok: bool, detail: str) -> None:
	status = 'PASS' if ok else 'FAIL'
	print(f'  {status}  {label:32} {detail}')
	if not ok:
		_failures.append(label)


def main() -> int:
	print('=== Retry loop verification (offline, fake LLM) ===\n')

	# Case 1: invalid -> invalid -> valid; loop must retry and recover.
	fake, result = _run([INVALID_EMPTY_NODE, INVALID_JUNK, VALID], max_retries=3)
	feedback_fed_back = all('model_validate_json' in t for t in fake.last_texts[1:3])
	check(
		'Case 1 invalid,invalid,valid',
		result is not None and fake.calls == 3 and fake.message_counts == [2, 4, 6] and feedback_fed_back,
		f'(calls={fake.calls}, msg_counts={fake.message_counts}, recovered={result is not None}, feedback={feedback_fed_back})',
	)

	# Case 2: all invalid; loop must exhaust max_retries and give up (None).
	fake, result = _run([INVALID_EMPTY_NODE, INVALID_EMPTY_NODE, INVALID_JUNK], max_retries=3)
	check(
		'Case 2 all invalid -> give up',
		result is None and fake.calls == 3 and fake.message_counts == [2, 4, 6],
		f'(calls={fake.calls}, msg_counts={fake.message_counts}, result={result})',
	)

	# Case 3: valid on first try; loop must NOT retry (guards against always-retry regression).
	fake, result = _run([VALID], max_retries=3)
	check(
		'Case 3 valid first try',
		result is not None and fake.calls == 1 and fake.message_counts == [2],
		f'(calls={fake.calls}, msg_counts={fake.message_counts}, recovered={result is not None})',
	)

	print()
	if _failures:
		print(f'{len(_failures)} case(s) FAILED: {", ".join(_failures)}')
		return 1
	print('All cases passed — the retry loop fires, feeds the pydantic error back, recovers, and gives up.')
	return 0


if __name__ == '__main__':
	try:
		sys.exit(main())
	finally:
		try:
			os.remove(CACHE_PATH)
		except OSError:
			pass

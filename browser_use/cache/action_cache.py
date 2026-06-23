"""Action caching layer for browser-use agent runs.

Records every executed action to a JSONL file so that a future deterministic run
can replay the *effective* actions without invoking the LLM. This module is purely
additive logging — it never changes agent behavior.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger('browser_use.cache')


def _classify_verdict(eval_text: str | None) -> str:
	"""Classify an eval_previous_goal string as 'success' / 'failure' / 'unknown'.

	Mirrors the agent's own log_response logic (success checked before failure) so the
	cache's notion of a verdict matches what the agent prints to the console.
	"""
	if not eval_text:
		return 'unknown'
	low = eval_text.lower()
	if 'success' in low:
		return 'success'
	if 'failure' in low:
		return 'failure'
	return 'unknown'


class ActionCache:
	"""Append-only JSONL recorder for executed agent actions.

	One instance maps to one agent run. Each executed action is written as a single
	JSON line, flushed immediately so partial runs still leave usable data behind.
	"""

	def __init__(self, cache_dir: str = '/Users/vasu/Desktop/Projects/browser-use/cached_memory') -> None:
		self.cache_dir = cache_dir
		os.makedirs(self.cache_dir, exist_ok=True)

		# Timestamp in a filesystem-safe form, e.g. cache_2026-06-21T04-35-11.jsonl
		timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
		self.jsonl_path = os.path.join(self.cache_dir, f'cache_{timestamp}.jsonl')
		self.summary_path = os.path.join(self.cache_dir, f'cache_{timestamp}.summary.json')

		self._start_time = datetime.now()
		self._records: list[dict[str, Any]] = []
		# Tracks (action_type, index, text-or-button) tuples already seen, for is_duplicate.
		self._seen_keys: set[tuple[Any, Any, Any]] = set()

		# Open the JSONL file for appending and keep the handle for fast flushed writes.
		self._file = open(self.jsonl_path, 'a', encoding='utf-8')

		logger.info(f'📝 Action cache created at {self.jsonl_path}')
		print(f'[ACTION_CACHE] Cache file created: {self.jsonl_path}')

	def _duplicate_key(self, action_data: dict[str, Any]) -> tuple[Any, Any, Any]:
		"""Build the identity tuple used for duplicate detection.

		Two actions are considered duplicates when they share the same action type,
		element index, and the relevant payload (text for inputs, button for clicks,
		text for done actions).
		"""
		action_type = action_data.get('action_type')
		index = action_data.get('index')
		# text covers input/done actions, button covers clicks; only one is present.
		payload = action_data.get('text', action_data.get('button'))
		return (action_type, index, payload)

	def record_action(self, action_data: dict[str, Any]) -> None:
		"""Append one action record to the JSONL file, flushing immediately.

		The caller provides the core fields (step_number, action_index_in_step,
		action_type, index, text/button, element_attributes, url). This method stamps
		the record with a timestamp and an is_duplicate flag before persisting it.
		"""
		record = dict(action_data)
		record.setdefault('timestamp', datetime.now().isoformat(timespec='milliseconds'))

		key = self._duplicate_key(record)
		record['is_duplicate'] = key in self._seen_keys
		self._seen_keys.add(key)

		self._records.append(record)
		self._file.write(json.dumps(record) + '\n')
		self._file.flush()
		print(
			f'[ACTION_CACHE] Recorded: step={record.get("step_number")} action={record.get("action_type")} '
			f'index={record.get("index")} dup={record.get("is_duplicate")}'
		)

	def finalize(self) -> None:
		"""Write a summary JSON file alongside the JSONL and close the handle."""
		total_steps = len({r.get('step_number') for r in self._records})
		redundant_count = sum(1 for r in self._records if r.get('is_duplicate'))
		unique_actions = len(self._records) - redundant_count
		total_duration = (datetime.now() - self._start_time).total_seconds()

		# Count distinct steps whose own eval_previous_goal carried a Failure verdict
		# (the agent judged the prior step's actions a dead end). One verdict per step.
		step_verdicts: dict[Any, str] = {}
		for r in self._records:
			step_verdicts.setdefault(r.get('step_number'), _classify_verdict(r.get('eval_previous_goal')))
		failure_verdict_steps = sum(1 for v in step_verdicts.values() if v == 'failure')

		summary = {
			'jsonl_path': self.jsonl_path,
			'total_actions': len(self._records),
			'total_steps': total_steps,
			'unique_actions': unique_actions,
			'redundant_count': redundant_count,
			'failure_verdict_steps': failure_verdict_steps,
			'total_duration_seconds': round(total_duration, 3),
		}

		with open(self.summary_path, 'w', encoding='utf-8') as f:
			json.dump(summary, f, indent=2)

		try:
			self._file.close()
		except Exception:
			pass

		logger.info(
			f'✅ Action cache finalized: {summary["total_actions"]} actions across {total_steps} steps '
			f'({redundant_count} redundant) → {self.summary_path}'
		)
		print(f'[ACTION_CACHE] Finalized: {summary}')

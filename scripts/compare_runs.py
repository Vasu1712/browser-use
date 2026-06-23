#!/usr/bin/env python3
"""
Log-based metrics for optexity automation runs, plus a convergence-table renderer for the
iterative optimizer (scripts/iterative_optimization.py).

Two responsibilities:
  1. Parse a single optexity run log into per-node outcomes + run metrics (how many LLM
     calls happened, how many nodes failed / fell back to the LLM, final status).
  2. Render a convergence table across iterations, showing LLM calls dropping toward zero
     as nodes become deterministic.

CLI:
    python compare_runs.py metrics <run.log> [--nodes N]
    python compare_runs.py converge <iterations.json>     # list[ {iteration, llm_calls, ...} ]

Pure standard library. The log markers mirror optexity's own logging:
  -----Running node new K-----            (run_automation.py)
  -----Finished node K-----               (run_automation.py)
  <Action> failed after N tries           (handle_command.py)  -> deterministic node failed
  Executing prompt-based action: <Action> (handle_input.py)    -> fell back to the LLM
  <Action> successful on try N            (handle_command.py)
  Next goal:                              (browser_use agent)  -> one per agentic LLM step
  Task <id> completed with status <s>     (run_automation.py)
"""

from __future__ import annotations

import json
import re
import sys

NODE_START = re.compile(r'-----Running node new (\d+)-----')
NODE_FINISH = re.compile(r'-----Finished node (\d+)-----')
CMD_FAILED = re.compile(r'(\w+) failed after \d+ tries')
LLM_FALLBACK = re.compile(r'Executing prompt-based action: (\w+)')
LOC_NOT_VISIBLE = re.compile(r'error: locator not visible')
AGENTIC_STEP = re.compile(r'Next goal:')  # emoji-prefixed in logs; substring is stable
RUN_STATUS = re.compile(r'completed with status (\w+)')


def parse_node_outcomes(log_text: str) -> list[dict]:
	"""Per-node pass/fail derived from the node markers and failure signals between them.

	A node is 'fail' if its command exhausted retries ("failed after N tries") OR it fell
	back to the LLM ("Executing prompt-based action") — both mean the deterministic locator
	did not work. "locator not visible" alone is a transient retry signal and does not by
	itself fail the node.
	"""
	nodes: dict[int, dict] = {}
	current: int | None = None
	for line in log_text.splitlines():
		m = NODE_START.search(line)
		if m:
			current = int(m.group(1))
			nodes.setdefault(current, {'hard_failure': False, 'fell_back': False, 'signals': []})
			continue
		m = NODE_FINISH.search(line)
		if m:
			current = None
			continue
		if current is None:
			continue
		if CMD_FAILED.search(line):
			nodes[current]['hard_failure'] = True
			nodes[current]['signals'].append('command_failed')
		elif LLM_FALLBACK.search(line):
			nodes[current]['fell_back'] = True
			nodes[current]['signals'].append('llm_fallback')
		elif LOC_NOT_VISIBLE.search(line):
			nodes[current]['signals'].append('locator_not_visible')

	outcomes = []
	for idx in sorted(nodes):
		o = nodes[idx]
		failed = o['hard_failure'] or o['fell_back']
		outcomes.append(
			{
				'index': idx,
				'status': 'fail' if failed else 'pass',
				'fell_back_to_llm': o['fell_back'],
				'signals': o['signals'],
			}
		)
	return outcomes


def extract_run_metrics(log_text: str) -> dict:
	"""Aggregate run-level metrics from a log.

	llm_calls = agentic steps (one LLM call each) + LLM fallbacks inside deterministic nodes.
	This is the number that must trend to zero as the automation becomes fully deterministic.
	"""
	outcomes = parse_node_outcomes(log_text)
	agentic_steps = len(AGENTIC_STEP.findall(log_text))
	llm_fallbacks = len(LLM_FALLBACK.findall(log_text))
	status_match = RUN_STATUS.search(log_text)
	return {
		'llm_calls': agentic_steps + llm_fallbacks,
		'agentic_steps': agentic_steps,
		'llm_fallbacks': llm_fallbacks,
		'nodes_total': len(outcomes),
		'nodes_failed': sum(1 for o in outcomes if o['status'] == 'fail'),
		'nodes_passed': sum(1 for o in outcomes if o['status'] == 'pass'),
		'status': status_match.group(1) if status_match else 'unknown',
		'node_outcomes': outcomes,
	}


def print_convergence_table(iterations: list[dict]) -> None:
	"""Render the per-iteration convergence table.

	Each iteration dict should carry: iteration, llm_calls, wall_clock,
	deterministic_nodes, total_nodes, status.
	"""
	cols = ['iter', 'llm_calls', 'wall_clock_s', 'deterministic', 'total_nodes', 'status']
	widths = {c: len(c) for c in cols}
	rows = []
	for it in iterations:
		row = {
			'iter': str(it.get('iteration', '?')),
			'llm_calls': str(it.get('llm_calls', '?')),
			'wall_clock_s': f'{it.get("wall_clock", 0):.1f}',
			'deterministic': str(it.get('deterministic_nodes', '?')),
			'total_nodes': str(it.get('total_nodes', '?')),
			'status': str(it.get('status', '?')),
		}
		rows.append(row)
		for c in cols:
			widths[c] = max(widths[c], len(row[c]))

	def fmt(values: dict) -> str:
		return '  '.join(str(values[c]).ljust(widths[c]) for c in cols)

	print('Convergence:')
	print('  ' + fmt({c: c for c in cols}))
	print('  ' + '  '.join('-' * widths[c] for c in cols))
	for row in rows:
		print('  ' + fmt(row))

	if iterations:
		first, last = iterations[0], iterations[-1]
		print(
			f'\n  LLM calls: {first.get("llm_calls", "?")} -> {last.get("llm_calls", "?")}  |  '
			f'deterministic nodes: {first.get("deterministic_nodes", "?")}/{first.get("total_nodes", "?")} '
			f'-> {last.get("deterministic_nodes", "?")}/{last.get("total_nodes", "?")}'
		)


def _cli_metrics(args: list[str]) -> int:
	if not args:
		print('usage: compare_runs.py metrics <run.log>', file=sys.stderr)
		return 2
	with open(args[0], encoding='utf-8', errors='replace') as f:
		log_text = f.read()
	metrics = extract_run_metrics(log_text)
	print(json.dumps(metrics, indent=2))
	return 0


def _cli_converge(args: list[str]) -> int:
	if not args:
		print('usage: compare_runs.py converge <iterations.json>', file=sys.stderr)
		return 2
	with open(args[0], encoding='utf-8') as f:
		iterations = json.load(f)
	print_convergence_table(iterations)
	return 0


def main() -> int:
	if len(sys.argv) < 2 or sys.argv[1] not in ('metrics', 'converge'):
		print('usage: compare_runs.py {metrics <run.log> | converge <iterations.json>}', file=sys.stderr)
		return 2
	if sys.argv[1] == 'metrics':
		return _cli_metrics(sys.argv[2:])
	return _cli_converge(sys.argv[2:])


if __name__ == '__main__':
	sys.exit(main())

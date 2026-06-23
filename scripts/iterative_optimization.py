#!/usr/bin/env python3
"""
Iterative self-healing optimizer: turn a single agentic optexity task into a fully
deterministic automation, healing failures across iterations.

Each iteration writes the current automation to OPTEXITY_AUTOMATION_PATH, triggers an
optexity run, observes per-node outcomes (from this run's browser-use cache for agentic
nodes, and from the optexity log for deterministic nodes), then rebuilds the automation:

  - An agentic node that ran -> its cache is converted into deterministic command nodes
    (reusing the rule-based scripts/cache_to_automation.py, so NO LLM runs at rebuild time).
  - A deterministic node that passed -> kept as-is.
  - A deterministic node that FAILED (locator missing / fell back to the LLM) -> reverted to
    a SCOPED agentic_task node whose task text is that step's captured next_goal, so the next
    iteration re-explores just that step and re-caches a working locator.

The LLM therefore runs only during agentic exploration inside optexity — never on a fully
deterministic replay, and never in this orchestrator (with the default rule-based synthesizer).

The run execution is dependency-injected: `run_fn(automation, iteration) -> RunResult`.
  - Live mode wires `run_fn` to write+trigger+wait against a real optexity server.
  - `simulate` mode injects a scripted run_fn to demonstrate convergence with no optexity.

CLI:
    python iterative_optimization.py run <initial_automation.json> [--max-iterations N] [...]
    python iterative_optimization.py simulate            # offline arxiv demo, no optexity

Standard library only (+ the repo's own scripts). LLM synthesizer is opt-in (--synthesizer llm).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_runs  # noqa: E402
from cache_to_automation import build_node, effective_actions  # noqa: E402

DEFAULT_CACHE_DIR = '/Users/vasu/Desktop/Projects/browser-use/cached_memory'
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_OSCILLATION_THRESHOLD = 3  # after this many pass/fail flips, a node stays agentic
DEFAULT_AGENTIC_MAX_STEPS = 8
DEFAULT_COMPLETION_TIMEOUT = 600.0  # seconds to wait for an optexity run to finish
DEFAULT_POLL_INTERVAL = 2.0


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass
class Config:
	automation_path: str = os.environ.get('OPTEXITY_AUTOMATION_PATH', '')
	log_path: str = os.environ.get('OPTEXITY_LOG_PATH', '')
	endpoint: str = os.environ.get('OPTEXITY_ENDPOINT', '')
	cache_dir: str = os.environ.get('OPTEXITY_CACHE_DIR', DEFAULT_CACHE_DIR)
	trigger_cmd: str = os.environ.get('OPTEXITY_TRIGGER_CMD', '')
	max_iterations: int = DEFAULT_MAX_ITERATIONS
	oscillation_threshold: int = DEFAULT_OSCILLATION_THRESHOLD
	agentic_max_steps: int = DEFAULT_AGENTIC_MAX_STEPS
	completion_timeout: float = DEFAULT_COMPLETION_TIMEOUT
	poll_interval: float = DEFAULT_POLL_INTERVAL
	synthesizer: str = 'rule'  # 'rule' (no LLM) | 'llm'
	output_path: str = 'converged_automation.json'


@dataclass
class NodeSpec:
	"""One automation node plus the orchestrator metadata needed to heal it."""

	mode: str  # 'agentic' | 'deterministic'
	node: dict
	goal: str = ''  # next_goal text — used to revert to a scoped agentic node
	flips: int = 0  # pass/fail flip counter (oscillation guard)
	frozen: bool = False  # permanently agentic after too many flips


@dataclass
class RunResult:
	log_text: str
	new_cache_files: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Pure helpers (composition, synthesis, policy) — all unit-testable without optexity
# --------------------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
	out = []
	with open(path, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line:
				out.append(json.loads(line))
	return out


def make_agentic_node(task: str, max_steps: int) -> dict:
	return {
		'type': 'action_node',
		'interaction_action': {'agentic_task': {'task': task or 'Complete this step.', 'max_steps': max_steps, 'backend': 'browser_use'}},
	}


def initial_specs(automation: dict) -> list[NodeSpec]:
	"""Wrap an initial automation's nodes as NodeSpecs (agentic or deterministic)."""
	specs = []
	for node in automation.get('nodes', []):
		ia = node.get('interaction_action', {})
		if 'agentic_task' in ia:
			specs.append(NodeSpec('agentic', node, goal=ia['agentic_task'].get('task', '')))
		else:
			specs.append(NodeSpec('deterministic', node, goal=''))
	return specs


def compose_automation(specs: list[NodeSpec], base: dict) -> dict:
	"""Build a full automation dict from the current specs, keeping base url/parameters."""
	return {
		'url': base.get('url', ''),
		'parameters': base.get('parameters', {'input_parameters': {}, 'generated_parameters': {}}),
		'nodes': [s.node for s in specs],
	}


def synthesize_specs_from_cache(cache_path: str, synthesizer: str = 'rule') -> list[NodeSpec]:
	"""Convert one agentic node's cache JSONL into deterministic NodeSpecs.

	Uses the rule-based converter's filtering + locator synthesis (no LLM). Each resulting
	node carries the originating step's next_goal so it can later be reverted to a scoped
	agentic node if its locator breaks.
	"""
	actions = load_jsonl(cache_path)
	specs: list[NodeSpec] = []
	for a in effective_actions(actions):
		node = build_node(a)
		if node:
			specs.append(NodeSpec('deterministic', node, goal=a.get('next_goal') or a.get('memory') or ''))
	return specs


def rebuild(specs: list[NodeSpec], outcomes: list[dict], new_cache_files: list[str], cfg: Config) -> list[NodeSpec]:
	"""Produce the next iteration's specs from this run's per-node outcomes + caches.

	- agentic spec that ran -> expand into deterministic specs synthesized from its cache.
	- deterministic spec that passed -> keep.
	- deterministic spec that failed -> revert to a scoped agentic node (using its goal),
      bumping the flip counter; freeze permanently after the oscillation threshold.
	"""
	outcome_by_index = {o['index']: o for o in outcomes}

	# New cache files map 1:1 to agentic nodes in execution order (each agentic_task node
	# spawns its own browser-use Agent -> its own cache file).
	agentic_positions = [i for i, s in enumerate(specs) if s.mode == 'agentic']
	caches_in_order = sorted(new_cache_files, key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else 0, p))
	cache_for_pos = dict(zip(agentic_positions, caches_in_order))

	new_specs: list[NodeSpec] = []
	for pos, s in enumerate(specs):
		if s.mode == 'agentic':
			cache = cache_for_pos.get(pos)
			synthesized = synthesize_specs_from_cache(cache, cfg.synthesizer) if cache else []
			if synthesized and not s.frozen:
				new_specs.extend(synthesized)
			else:
				new_specs.append(s)  # nothing synthesized, or frozen -> keep agentic
		else:
			outcome = outcome_by_index.get(pos)
			if outcome and outcome['status'] == 'fail':
				flips = s.flips + 1
				frozen = flips >= cfg.oscillation_threshold
				agentic = NodeSpec('agentic', make_agentic_node(s.goal, cfg.agentic_max_steps), goal=s.goal, flips=flips, frozen=frozen)
				new_specs.append(agentic)
			else:
				new_specs.append(s)
	return new_specs


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


class IterativeOptimizer:
	def __init__(self, cfg: Config, run_fn=None):
		self.cfg = cfg
		self.run_fn = run_fn or self._real_run

	def optimize(self, initial_automation: dict) -> dict:
		base = initial_automation
		specs = initial_specs(initial_automation)
		iterations: list[dict] = []
		converged = False
		reason = 'max_iterations'

		for i in range(1, self.cfg.max_iterations + 1):
			automation = compose_automation(specs, base)
			det_count = sum(1 for s in specs if s.mode == 'deterministic')
			total = len(specs)
			print(f'\n=== Iteration {i}: {det_count}/{total} deterministic nodes ===')

			t0 = time.time()
			result = self.run_fn(automation, i)
			wall = time.time() - t0

			metrics = compare_runs.extract_run_metrics(result.log_text)
			outcomes = metrics['node_outcomes']
			iterations.append(
				{
					'iteration': i,
					'llm_calls': metrics['llm_calls'],
					'wall_clock': wall,
					'deterministic_nodes': det_count,
					'total_nodes': total,
					'status': metrics['status'],
				}
			)
			print(
				f'  status={metrics["status"]} llm_calls={metrics["llm_calls"]} '
				f'nodes_failed={metrics["nodes_failed"]}/{metrics["nodes_total"]}'
			)

			all_det = all(s.mode == 'deterministic' for s in specs)
			if all_det and metrics['status'] == 'success' and metrics['nodes_failed'] == 0:
				converged = True
				reason = 'converged'
				break

			next_specs = rebuild(specs, outcomes, result.new_cache_files, self.cfg)

			# No-improvement guard: identical spec signature two iterations running.
			if _spec_signature(next_specs) == _spec_signature(specs) and i > 1:
				reason = 'no_improvement'
				specs = next_specs
				break
			specs = next_specs

		final = compose_automation(specs, base)
		self._write_output(final)
		print('\n' + '=' * 60)
		print(f'Result: {"CONVERGED" if converged else "stopped"} ({reason}) after {len(iterations)} iteration(s)')
		compare_runs.print_convergence_table(iterations)
		print(f'\nFinal automation -> {self.cfg.output_path}')
		return {'converged': converged, 'reason': reason, 'iterations': iterations, 'automation': final}

	def _write_output(self, automation: dict) -> None:
		with open(self.cfg.output_path, 'w', encoding='utf-8') as f:
			json.dump(automation, f, indent=2)

	# --- live run execution -----------------------------------------------------------

	def _real_run(self, automation: dict, iteration: int) -> RunResult:
		cfg = self.cfg
		if not cfg.automation_path:
			raise SystemExit('OPTEXITY_AUTOMATION_PATH is not set (where optexity reads the automation).')
		if not cfg.log_path:
			raise SystemExit('OPTEXITY_LOG_PATH is not set (the optexity run log to parse).')
		if not cfg.trigger_cmd:
			raise SystemExit('OPTEXITY_TRIGGER_CMD is not set (the shell command/curl that starts a run).')

		with open(cfg.automation_path, 'w', encoding='utf-8') as f:
			json.dump(automation, f, indent=2)

		before_summaries = _snapshot_summaries(cfg.cache_dir)
		before_offset = os.path.getsize(cfg.log_path) if os.path.exists(cfg.log_path) else 0

		print(f'  triggering run (iteration {iteration})...')
		proc = subprocess.run(cfg.trigger_cmd, shell=True, check=False, capture_output=True, text=True)
		task_id = _extract_task_id(proc.stdout)
		print(f'  task_id={task_id or "<none>"}; waiting for completion...')

		log_text, new_caches = _wait_for_completion(cfg, before_summaries, before_offset, task_id)
		return RunResult(log_text=log_text, new_cache_files=new_caches)


def _spec_signature(specs: list[NodeSpec]) -> tuple:
	return tuple((s.mode, json.dumps(s.node, sort_keys=True)) for s in specs)


def _snapshot_summaries(cache_dir: str) -> set[str]:
	return set(glob.glob(os.path.join(cache_dir, '*.summary.json')))


def _extract_task_id(trigger_stdout: str) -> str | None:
	"""Pull the task_id out of the trigger's response (optexity returns it as JSON)."""
	if not trigger_stdout:
		return None
	try:
		return json.loads(trigger_stdout.strip().splitlines()[-1]).get('task_id')
	except Exception:
		m = re.search(r'"task_id"\s*:\s*"([^"]+)"', trigger_stdout)
		return m.group(1) if m else None


def _wait_for_completion(
	cfg: Config, before_summaries: set[str], before_offset: int, task_id: str | None
) -> tuple[str, list[str]]:
	"""Poll the log from before_offset until THIS run finishes; return its slice + new caches.

	Completion is keyed off the run-specific line `Task <task_id> completed with status ...`
	when a task_id is known (avoids matching a stale/partial run), falling back to a generic
	`completed with status` otherwise. The cache summary is written before that line is
	logged, so any new *.summary.json present at completion belongs to this run.
	"""
	done_re = re.compile(rf'Task {re.escape(task_id)} completed with status') if task_id else re.compile(r'completed with status')
	deadline = time.time() + cfg.completion_timeout
	while time.time() < deadline:
		time.sleep(cfg.poll_interval)
		log_slice = ''
		if os.path.exists(cfg.log_path):
			with open(cfg.log_path, encoding='utf-8', errors='replace') as f:
				f.seek(before_offset)
				log_slice = f.read()
		if done_re.search(log_slice):
			new_summaries = _snapshot_summaries(cfg.cache_dir) - before_summaries
			new_caches = [s.replace('.summary.json', '.jsonl') for s in new_summaries]
			new_caches = [c for c in new_caches if os.path.exists(c)]
			return log_slice, new_caches
	raise TimeoutError(f'optexity run did not complete within {cfg.completion_timeout}s')


# --------------------------------------------------------------------------------------
# Offline simulation: the arxiv brittle-locator demo, no optexity required
# --------------------------------------------------------------------------------------


def _simulate_arxiv() -> dict:
	"""Drive the optimizer through the arxiv scenario with a scripted run_fn.

	Iter 1: one agentic node -> caches 4 steps -> 4 deterministic nodes.
	Iter 2: deterministic run; the arXiv-id link node (idx 2) breaks (paper changed) -> the
	        run falls back to the LLM for that node -> node 2 reverted to scoped agentic.
	Iter 3: node 2 agentic re-caches the (new) first paper -> re-synthesized deterministic.
	Iter 4: all deterministic, clean run -> converged.
	"""
	import tempfile

	tmp = tempfile.mkdtemp(prefix='itersim_')

	def write_cache(name: str, rows: list[dict]) -> str:
		path = os.path.join(tmp, name)
		with open(path, 'w', encoding='utf-8') as f:
			f.write('\n'.join(json.dumps(r) for r in rows) + '\n')
		return path

	# Iter 1 cache: the full agentic exploration (4 effective actions, each with a next_goal).
	iter1_cache = write_cache(
		'cache_iter1.jsonl',
		[
			{'step_number': 1, 'action_index_in_step': 0, 'action_type': 'input_text', 'index': 21, 'text': 'AI', 'url': 'https://arxiv.org/', 'is_duplicate': False, 'next_goal': "Input 'AI' into the search box and click Search.", 'element_attributes': {'tag': 'input', 'name': 'query'}},
			{'step_number': 1, 'action_index_in_step': 1, 'action_type': 'click', 'index': 186, 'button': 'left', 'url': 'https://arxiv.org/', 'is_duplicate': False, 'next_goal': "Input 'AI' into the search box and click Search.", 'element_attributes': {'tag': 'button', 'visible_text': 'Search', 'role': 'button'}},
			{'step_number': 3, 'action_index_in_step': 0, 'action_type': 'click', 'index': 1914, 'button': 'left', 'url': 'https://arxiv.org/search/?query=AI', 'is_duplicate': False, 'next_goal': "Click on the first paper's arXiv link to open its details page.", 'element_attributes': {'tag': 'a', 'visible_text': 'arXiv:2606.20539', 'role': 'link'}},
			{'step_number': 5, 'action_index_in_step': 0, 'action_type': 'click', 'index': 7910, 'button': 'left', 'url': 'https://arxiv.org/abs/2606.20539', 'is_duplicate': False, 'next_goal': "Click on the 'View PDF' link to download the paper's PDF.", 'element_attributes': {'tag': 'a', 'visible_text': 'View PDF', 'role': 'link'}},
		],
	)

	# Iter 3 cache: re-exploring just the failed step; the NEW top paper has a different id.
	iter3_cache = write_cache(
		'cache_iter3.jsonl',
		[
			{'step_number': 1, 'action_index_in_step': 0, 'action_type': 'click', 'index': 1920, 'button': 'left', 'url': 'https://arxiv.org/search/?query=AI', 'is_duplicate': False, 'next_goal': "Click on the first paper's arXiv link to open its details page.", 'element_attributes': {'tag': 'a', 'visible_text': 'arXiv:2607.11111', 'role': 'link'}},
		],
	)

	def log_agentic(steps: int) -> str:
		lines = ['-----Running node new 0-----']
		for k in range(steps):
			lines.append(f'  Next goal: step {k}')
		lines += ['-----Finished node 0-----', 'Task abc completed with status success']
		return '\n'.join(lines)

	def log_deterministic(fail_index: int | None, n_nodes: int) -> str:
		lines = []
		for k in range(n_nodes):
			lines.append(f'-----Running node new {k}-----')
			if k == fail_index:
				lines.append('  ClickElementAction failed after 10 tries: error: locator not visible')
				lines.append('  Executing prompt-based action: ClickElementAction')  # LLM fallback (heals at runtime)
			else:
				lines.append('  InputTextAction successful on try 1' if k == 0 else '  ClickElementAction successful on try 1')
			lines.append(f'-----Finished node {k}-----')
		lines.append('Task abc completed with status success')
		return '\n'.join(lines)

	scripted = {
		1: RunResult(log_agentic(7), [iter1_cache]),
		2: RunResult(log_deterministic(fail_index=2, n_nodes=4), []),
		3: RunResult(log_agentic(1).replace('node new 0', 'node new 2').replace('Finished node 0', 'Finished node 2'), [iter3_cache]),
		4: RunResult(log_deterministic(fail_index=None, n_nodes=4), []),
	}

	def fake_run(automation: dict, iteration: int) -> RunResult:
		return scripted.get(iteration, RunResult(log_deterministic(None, len(automation['nodes'])), []))

	initial = {
		'url': 'https://arxiv.org/',
		'parameters': {'input_parameters': {'search_query': ['AI']}, 'generated_parameters': {}},
		'nodes': [make_agentic_node("Search for 'AI', open the first paper, and download its PDF.", 15)],
	}

	cfg = Config(max_iterations=6, output_path=os.path.join(tmp, 'converged_arxiv.json'))
	opt = IterativeOptimizer(cfg, run_fn=fake_run)
	return opt.optimize(initial)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
	parser = argparse.ArgumentParser(description='Iterative self-healing automation optimizer.')
	sub = parser.add_subparsers(dest='cmd', required=True)

	p_run = sub.add_parser('run', help='Optimize against a live optexity server.')
	p_run.add_argument(
		'initial', nargs='?', default=None, help='Initial automation JSON (positional, or use --input).'
	)
	p_run.add_argument('--input', dest='input_path', default=None, help='Initial automation JSON (alias for the positional arg).')
	p_run.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS)
	p_run.add_argument('--synthesizer', choices=['rule', 'llm'], default='rule')
	p_run.add_argument('--output', default='converged_automation.json')
	p_run.add_argument('--oscillation-threshold', type=int, default=DEFAULT_OSCILLATION_THRESHOLD)

	sub.add_parser('simulate', help='Offline arxiv convergence demo (no optexity needed).')

	args = parser.parse_args()

	if args.cmd == 'simulate':
		result = _simulate_arxiv()
		return 0 if result['converged'] else 1

	cfg = Config(
		max_iterations=args.max_iterations,
		synthesizer=args.synthesizer,
		output_path=args.output,
		oscillation_threshold=args.oscillation_threshold,
	)
	initial_path = args.input_path or args.initial
	if not initial_path:
		print('error: provide the initial automation as a positional arg or via --input', file=sys.stderr)
		return 2
	with open(initial_path, encoding='utf-8') as f:
		initial = json.load(f)
	result = IterativeOptimizer(cfg).optimize(initial)
	return 0 if result['converged'] else 1


if __name__ == '__main__':
	sys.exit(main())

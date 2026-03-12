#!/usr/bin/env python3
"""Behavior cloning + optional RL fine-tuning for Phase F controller policies.

先模仿一个已知好的 heuristic controller，再选择是否做少量 RL-like 微调。
This tests whether the current difficulty comes from reward optimization or from
controller representation / policy class mismatch.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from phase_f_controller_policy_sweep import ExampleTrace, load_example_traces, simulate_policy
from phase_f_controller_generator_robustness import load_generator_map
from phase_f_train_trainable_controller import (
    LinearPolicy,
    build_features,
    compute_class_weights,
    evaluate_policy,
    split_traces,
    train_policy,
)


def teacher_actions(trace: ExampleTrace, family: str, params: dict[str, Any]) -> list[tuple[list[float], float]]:
    """Generate per-step stop labels from one heuristic teacher policy."""

    sims = simulate_policy(trace, family, params)
    stopped_at = sims["steps_processed"] - 1 if sims["predicted_erroneous"] else None
    rows = []
    for idx, _row in enumerate(trace.rows):
        label = 1.0 if (stopped_at is not None and idx == stopped_at) else 0.0
        rows.append((build_features(trace, idx), label))
        if stopped_at is not None and idx >= stopped_at:
            break
    return rows


def train_behavior_clone(
    train_traces: list[ExampleTrace],
    dev_traces: list[ExampleTrace],
    generator_map: dict[str, str],
    *,
    teacher_family: str,
    teacher_params: dict[str, Any],
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
) -> tuple[LinearPolicy, list[dict[str, Any]], dict[str, Any]]:
    """Supervised behavior cloning from a heuristic teacher."""

    policy = LinearPolicy(input_dim=10, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    class_weights = compute_class_weights(train_traces)
    curve = []
    best_score = -1e9
    best_state = None
    best_eval = None

    train_pairs = []
    for trace in train_traces:
        train_pairs.extend(teacher_actions(trace, teacher_family, teacher_params))
    if not train_pairs:
        raise RuntimeError('No train_pairs for behavior cloning')

    x = torch.tensor([f for f, _y in train_pairs], dtype=torch.float32)
    y = torch.tensor([yy for _f, yy in train_pairs], dtype=torch.float32)
    pos_weight = torch.tensor([(len(train_pairs) - float(y.sum().item())) / max(float(y.sum().item()), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        opt.zero_grad()
        logits = policy.net(x).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()

        train_eval = evaluate_policy(train_traces, policy, generator_map, reward_mode='balanced', class_weights=class_weights)
        dev_eval = evaluate_policy(dev_traces, policy, generator_map, reward_mode='balanced', class_weights=class_weights) if dev_traces else train_eval
        score = dev_eval['metrics']['balanced_f1']
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            best_eval = dev_eval
        curve.append({
            'epoch': epoch,
            'loss': float(loss.detach().item()),
            'train_balanced_f1': train_eval['metrics']['balanced_f1'],
            'dev_balanced_f1': dev_eval['metrics']['balanced_f1'],
        })

    assert best_state is not None and best_eval is not None
    policy.load_state_dict(best_state)
    return policy, curve, best_eval


def render_summary_md(run_dir: Path, summary: dict[str, Any]) -> str:
    lines = [
        '# Phase F Behavior Cloning / RL Fine-Tune',
        '',
        f"- run_dir: `{run_dir}`",
        '',
        '| case_id | mode | teacher | dev_balanced_f1 | test_balanced_f1 | test_worst_gen_f1 | test_step_frac |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for case in summary['cases']:
        worst = case['test_eval']['worst_generator']['metrics']['balanced_f1'] if case['test_eval']['worst_generator'] else None
        lines.append(
            '| {case} | {mode} | `{teacher}` | {dev:.4f} | {ev:.4f} | {worst} | {sf:.4f} |'.format(
                case=case['case_id'],
                mode=case['mode'],
                teacher=json.dumps(case['teacher'], sort_keys=True),
                dev=case['best_dev_eval']['metrics']['balanced_f1'],
                ev=case['test_eval']['metrics']['balanced_f1'],
                worst=(f'{worst:.4f}' if worst is not None else 'N/A'),
                sf=case['test_eval']['efficiency']['mean_step_fraction'],
            )
        )
    lines.extend([
        '',
        '- `bc_only` = 只模仿 heuristic teacher。',
        '- `bc_then_rl` = 先模仿，再做少量 REINFORCE 微调。',
        '- 主结果固定看 `test_eval`，`full_eval` 仅作为 in-benchmark 上界参考。',
        '',
    ])
    return '\n'.join(lines) + '\n'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Behavior clone a heuristic controller and optionally fine-tune with RL-like updates.')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--case', action='append', default=[], help='CASE_ID|SCORED_ROWS|PROCESSBENCH_JSON|TEACHER_FAMILY|TEACHER_PARAMS_JSON')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--bc-epochs', type=int, default=50)
    parser.add_argument('--bc-learning-rate', type=float, default=3e-3)
    parser.add_argument('--do-rl-finetune', action='store_true')
    parser.add_argument('--rl-epochs', type=int, default=40)
    parser.add_argument('--rl-learning-rate', type=float, default=1e-3)
    parser.add_argument('--robust-lambda', type=float, default=0.0)
    parser.add_argument('--dev-fraction', type=float, default=0.2)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit('At least one --case is required.')

    run_dir = Path('assets/artifacts/phase_f_bc') / (f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cases_out = []
    for raw_case in args.case:
        case_id, scored_rows_path, processbench_json, teacher_family, teacher_params_json = raw_case.split('|', 4)
        teacher_params = json.loads(teacher_params_json)
        traces = load_example_traces(Path(scored_rows_path), fallback_benchmark_id=case_id)
        generator_map = load_generator_map(Path(processbench_json))
        train_traces, dev_traces, test_traces = split_traces(
            traces,
            generator_map,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
        )
        policy, curve, best_dev_eval = train_behavior_clone(
            train_traces,
            dev_traces,
            generator_map,
            teacher_family=teacher_family,
            teacher_params=teacher_params,
            hidden_dim=args.hidden_dim,
            learning_rate=args.bc_learning_rate,
            epochs=args.bc_epochs,
        )
        mode = 'bc_only'
        if args.do_rl_finetune:
            policy, rl_curve, best_dev_eval = train_policy(
                train_traces,
                dev_traces,
                generator_map,
                seed=args.seed,
                epochs=args.rl_epochs,
                learning_rate=args.rl_learning_rate,
                hidden_dim=args.hidden_dim,
                robust_lambda=args.robust_lambda,
                selection_metric='worst_generator_balanced_f1' if args.robust_lambda > 0 else 'balanced_f1',
                reward_mode='balanced',
                init_policy=policy,
            )
            curve.extend({'phase': 'rl', **row} for row in rl_curve)
            mode = 'bc_then_rl'
        else:
            curve = [{'phase': 'bc', **row} for row in curve]
        test_eval = evaluate_policy(
            test_traces if test_traces else traces,
            policy,
            generator_map,
            reward_mode='balanced',
            class_weights=compute_class_weights(train_traces),
        )
        full_eval = evaluate_policy(
            traces,
            policy,
            generator_map,
            reward_mode='balanced',
            class_weights=compute_class_weights(train_traces),
        )
        torch.save(policy.state_dict(), run_dir / f'{case_id}_policy.pt')
        (run_dir / f'{case_id}_curve.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in curve), encoding='utf-8')
        cases_out.append({
            'case_id': case_id,
            'mode': mode,
            'teacher': {'family': teacher_family, 'params': teacher_params},
            'num_train': len(train_traces),
            'num_dev': len(dev_traces),
            'num_test': len(test_traces),
            'best_dev_eval': best_dev_eval,
            'test_eval': test_eval,
            'full_eval': full_eval,
            'evaluation_scope': 'benchmark_internal_train_dev_test_split',
            'full_eval_warning': 'full_eval reuses training-family traces and must not be treated as external generalization.',
        })
        print(f"{case_id:>16} | mode={mode:<10} | dev={best_dev_eval['metrics']['balanced_f1']:.4f} | test={test_eval['metrics']['balanced_f1']:.4f}")
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(run_dir),
        'dev_fraction': args.dev_fraction,
        'test_fraction': args.test_fraction,
        'evaluation_scope': 'benchmark_internal_train_dev_test_split',
        'scope_warning': 'test_eval is the main metric; full_eval is in-benchmark and should not be used as external generalization evidence.',
        'cases': cases_out,
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (run_dir / 'summary.md').write_text(render_summary_md(run_dir, summary), encoding='utf-8')
    print(f'summary_json: {run_dir / "summary.json"}')
    print(f'summary_md: {run_dir / "summary.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

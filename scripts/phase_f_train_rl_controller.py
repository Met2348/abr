#!/usr/bin/env python3
"""Phase F F3: RL-trained ABR-lite controller (offline REINFORCE).

离线 REINFORCE 训练 ABR-lite stop/continue 控制器。
在不调用任何新推理的情况下，仅凭已打分的 ProcessBench prefix 轨迹训练神经网络策略。

Goal: learn a stop policy π(a_t | s_t) that achieves higher binary-detection F1
than the best heuristic (threshold_only at tau=0.38, F1≈0.867) while keeping
average compute fraction low.

MDP formulation
---------------
- State s_t: feature vector derived from score history up to step t
- Action a_t ∈ {0=continue, 1=stop-and-flag-error}
- Episode terminates when a_t = 1 (stop) or trace is exhausted (→ predict all-correct)
- Reward R:
    - Erroneous example (label k ≥ 0):
        stop at step t ≤ k  → +1.0  (correct early flag)
        stop at step t > k  →  0.0  (flagged but too late; not counted as binary correct)
        exhaust trace       → -1.0  (missed error)
    - All-correct example (label -1):
        stop at any step    → -1.0  (false alarm)
        exhaust trace       →  +1.0 (correct)
    - Efficiency bonus (optional): +alpha * (1 - steps/total) when reward > 0

Policy families
---------------
1. linear:   sigmoid(w · features)
2. mlp:      2-layer ReLU MLP on features
3. gru:      GRU processes scores one by one; linear readout at each step

Features (for linear/mlp):
  score_t, score_t-1 (or 0), delta_1, delta_2, running_mean, running_min, step_frac

Usage
-----
python scripts/phase_f_train_rl_controller.py \\
    --scored-rows assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_math_fulleval_20260311T123421Z/scored_rows.jsonl \\
    --output-dir assets/artifacts/phase_f_rl_controller/pbr19_math_v1 \\
    --arch mlp --hidden-dim 64 --lr 3e-3 --epochs 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ScoreTrace:
    """One ProcessBench example as a list of step scores.

    label=-1 → all-correct; label=k → first bad step index k.
    """

    example_id: str
    label: int
    scores: list[float] = field(default_factory=list)

    @property
    def is_all_correct(self) -> bool:
        return self.label == -1

    @property
    def n_steps(self) -> int:
        return len(self.scores)


def load_traces(scored_rows_jsonl: Path) -> list[ScoreTrace]:
    by_example: dict[str, ScoreTrace] = {}
    for line in scored_rows_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = row["example_id"]
        if eid not in by_example:
            by_example[eid] = ScoreTrace(example_id=eid, label=row["label"])
        by_example[eid].scores.append((row["prefix_step_index"], row["score"]))
    traces = []
    for tr in by_example.values():
        tr.scores.sort(key=lambda x: x[0])
        tr.scores = [s for _, s in tr.scores]
        traces.append(tr)
    return traces


def split_traces(
    traces: list[ScoreTrace],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[ScoreTrace], list[ScoreTrace], list[ScoreTrace]]:
    rng = random.Random(seed)
    shuffled = traces.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def make_features(
    scores_seen: list[float],
    *,
    max_steps: int,
    window: int = 4,
) -> torch.Tensor:
    """Convert score history (up to current step) to a fixed-dim feature vector.

    Features (dim = 2*window + 5):
      - last `window` scores (padded with 1.0 if not enough history)
      - first-differences of last `window` scores (padded with 0.0)
      - running_mean, running_min, running_max
      - step_frac = len(scores_seen) / max_steps
      - n_below_05 fraction (fraction of scores seen < 0.5)
    """
    t = len(scores_seen)
    # Pad scores window
    padded = [1.0] * (window - t) + scores_seen[-window:] if t < window else scores_seen[-window:]
    # First differences
    diffs = [padded[i] - padded[i - 1] for i in range(1, window)]
    diffs = [0.0] + diffs  # keep same length

    mean_s = statistics.mean(scores_seen)
    min_s = min(scores_seen)
    max_s = max(scores_seen)
    step_frac = t / max_steps
    n_below = sum(1 for s in scores_seen if s < 0.5) / t

    feats = padded + diffs + [mean_s, min_s, max_s, step_frac, n_below]
    return torch.tensor(feats, dtype=torch.float32)


FEATURE_DIM_FOR_WINDOW = {4: 13, 6: 17, 8: 21}


def feature_dim(window: int = 4) -> int:
    return 2 * window + 5


# ---------------------------------------------------------------------------
# Policy networks
# ---------------------------------------------------------------------------


class LinearPolicy(nn.Module):
    """Linear policy: p(stop) = sigmoid(w · features + b)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x)).squeeze(-1)

    def stop_prob(self, features: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.forward(features.unsqueeze(0)))


class MLPPolicy(nn.Module):
    """2-layer MLP policy."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)

    def stop_prob(self, features: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.forward(features.unsqueeze(0)))


class GRUPolicy(nn.Module):
    """GRU-based recurrent policy.

    Processes score (1-dim) at each step; outputs stop prob via linear readout.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_size=2, hidden_size=hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self._h: torch.Tensor | None = None

    def reset(self) -> None:
        self._h = None

    def step(self, score: float, step_frac: float) -> float:
        """Process one step score; returns stop probability."""
        x = torch.tensor([[score, step_frac]], dtype=torch.float32)
        if self._h is None:
            self._h = torch.zeros(1, self.hidden_dim)
        self._h = self.gru(x, self._h)
        p_stop = torch.sigmoid(self.readout(self._h)).item()
        return float(p_stop)

    def step_batch(
        self,
        score_t: torch.Tensor,
        frac_t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch step for training. Returns (p_stop, h_new)."""
        x = torch.stack([score_t, frac_t], dim=-1)  # (B, 2)
        h_new = self.gru(x, h)
        p_stop = torch.sigmoid(self.readout(h_new)).squeeze(-1)
        return p_stop, h_new


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def compute_reward(
    trace: ScoreTrace,
    *,
    stopped_at: int | None,
    efficiency_alpha: float = 0.0,
    fp_penalty: float = 2.0,
) -> float:
    """Compute episodic reward.

    stopped_at=None means trace exhausted (predict all-correct).

    fp_penalty: multiplier for false-alarm penalty (>1 penalizes FP more than FN).
    Setting fp_penalty=2 makes "always stop" have negative expected reward,
    breaking the local optimum.
    """
    n = trace.n_steps
    if stopped_at is None:
        # Exhausted: predict all-correct
        base = 1.0 if trace.is_all_correct else -1.0
        eff = 0.0
    else:
        steps_used = stopped_at + 1
        if trace.is_all_correct:
            base = -fp_penalty  # false alarm (penalized more)
            eff = 0.0
        elif stopped_at <= trace.label:
            base = 1.0  # correct early flag
            eff = efficiency_alpha * (1.0 - steps_used / n) if n > 0 else 0.0
        else:
            # Stopped after first-bad step (missed error in time)
            base = -0.5
            eff = 0.0
    return base + eff


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------


def rollout_episode(
    trace: ScoreTrace,
    policy: LinearPolicy | MLPPolicy | GRUPolicy,
    *,
    window: int = 4,
    deterministic: bool = False,
    max_steps_cap: int = 32,
    training: bool = False,
) -> tuple[list[torch.Tensor], int | None]:
    """Run one episode.

    Returns:
      log_probs: list of log-probability tensors for each action taken (with grad_fn when training=True)
      stopped_at: step index where stop action was taken (None if exhausted)
    """
    max_steps = min(trace.n_steps, max_steps_cap)
    log_probs: list[torch.Tensor] = []
    stopped_at = None

    if isinstance(policy, GRUPolicy):
        policy.reset()

    scores_seen: list[float] = []

    if isinstance(policy, GRUPolicy):
        h = torch.zeros(1, policy.hidden_dim)

    for t, score in enumerate(trace.scores[:max_steps]):
        scores_seen.append(score)
        step_frac = (t + 1) / max_steps

        if isinstance(policy, GRUPolicy):
            score_t = torch.tensor([score], dtype=torch.float32)
            frac_t = torch.tensor([step_frac], dtype=torch.float32)
            p_stop_t, h = policy.step_batch(score_t, frac_t, h)
            p_stop = float(p_stop_t.detach())
        else:
            feats = make_features(scores_seen, max_steps=max_steps, window=window)
            if training:
                p_stop_t = policy(feats.unsqueeze(0))  # tensor with grad
            else:
                with torch.no_grad():
                    p_stop_t = policy(feats.unsqueeze(0))
            p_stop = float(p_stop_t.detach())

        # Sample or greedy action
        if deterministic:
            action = 1 if p_stop >= 0.5 else 0
        else:
            action = 1 if random.random() < p_stop else 0

        # Compute log_prob as tensor to retain grad_fn
        if isinstance(policy, GRUPolicy):
            score_t2 = torch.tensor([score], dtype=torch.float32)
            frac_t2 = torch.tensor([step_frac], dtype=torch.float32)
            # Re-run with requires-grad for log_prob (GRU hidden needs full re-trace)
            # For simplicity use p_stop_t from above (already computed)
            lp = torch.log(p_stop_t + 1e-9) if action == 1 else torch.log(1 - p_stop_t + 1e-9)
        else:
            feats = make_features(scores_seen, max_steps=max_steps, window=window)
            if training:
                p_t = policy(feats.unsqueeze(0))
            else:
                p_t = p_stop_t
            lp = torch.log(p_t + 1e-9) if action == 1 else torch.log(1 - p_t + 1e-9)

        log_probs.append(lp)

        if action == 1:
            stopped_at = t
            break

    return log_probs, stopped_at


# ---------------------------------------------------------------------------
# Supervised pre-training (Behavioral Cloning from heuristic oracle)
# ---------------------------------------------------------------------------


def make_bc_dataset(
    traces: list[ScoreTrace],
    *,
    oracle_tau: float = 0.38,
    window: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (features, labels) dataset from heuristic oracle decisions.

    For each step t in each trace, the oracle label is:
      1 (stop) if score < oracle_tau (and trace is erroneous)
      0 (continue) otherwise

    This creates balanced supervised signal for BC warmstart.
    """
    X, Y = [], []
    for trace in traces:
        n = trace.n_steps
        scores_seen: list[float] = []
        stopped = False
        for t, score in enumerate(trace.scores):
            scores_seen.append(score)
            feats = make_features(scores_seen, max_steps=n, window=window)

            # Oracle: stop if score < oracle_tau (regardless of correct/not)
            # But don't penalize stopping on all-correct if score is genuinely low
            oracle_stop = (score < oracle_tau)
            X.append(feats)
            Y.append(float(oracle_stop))

            if oracle_stop:
                stopped = True
                break

        # If exhausted without stopping: all-correct episode, no stop
    return torch.stack(X), torch.tensor(Y, dtype=torch.float32)


def pretrain_bc(
    policy: LinearPolicy | MLPPolicy,
    traces: list[ScoreTrace],
    *,
    oracle_tau: float = 0.38,
    window: int = 4,
    lr: float = 1e-2,
    epochs: int = 50,
    pos_weight: float = 2.0,
) -> None:
    """Pre-train policy with behavioral cloning from heuristic oracle."""
    X, Y = make_bc_dataset(traces, oracle_tau=oracle_tau, window=window)
    # class-weighted BCE
    weight_tensor = torch.where(Y > 0.5, torch.tensor(pos_weight), torch.tensor(1.0))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = policy(X)
        loss = F.binary_cross_entropy(preds, Y, weight=weight_tensor)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Supervised sequence classifier (alternative to REINFORCE)
# ---------------------------------------------------------------------------


def make_sequence_dataset(
    traces: list[ScoreTrace],
    *,
    window: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dataset where label_t = P(erroneous AND t >= first_bad_step).

    At each step t of erroneous trace: label = 1 if t >= trace.label
    At each step of all-correct trace: label = 0

    This trains the policy to estimate: "has the error been revealed by step t?"
    """
    X, Y = [], []
    for trace in traces:
        n = trace.n_steps
        scores_seen: list[float] = []
        for t, score in enumerate(trace.scores):
            scores_seen.append(score)
            feats = make_features(scores_seen, max_steps=n, window=window)
            if trace.is_all_correct:
                label = 0.0
            else:
                label = 1.0 if t >= trace.label else 0.0
            X.append(feats)
            Y.append(label)
    return torch.stack(X), torch.tensor(Y, dtype=torch.float32)


def train_supervised(
    policy: nn.Module,
    traces: list[ScoreTrace],
    val_traces: list[ScoreTrace],
    *,
    window: int = 4,
    lr: float = 3e-3,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int = 30,
    pos_weight: float = 1.5,
) -> dict[str, Any]:
    """Train policy with supervised BCE on sequence step labels."""
    X_train, Y_train = make_sequence_dataset(traces, window=window)
    X_val, Y_val = make_sequence_dataset(val_traces, window=window)

    pw = torch.tensor([pos_weight])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        policy.train()
        perm = torch.randperm(len(X_train))
        X_train, Y_train = X_train[perm], Y_train[perm]

        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i : i + batch_size]
            yb = Y_train[i : i + batch_size]
            optimizer.zero_grad()
            pred = policy(xb)
            w = torch.where(yb > 0.5, pw, torch.ones_like(yb))
            loss = F.binary_cross_entropy(pred, yb, weight=w)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        policy.eval()
        with torch.no_grad():
            val_pred = policy(X_val)
            w_val = torch.where(Y_val > 0.5, pw, torch.ones_like(Y_val))
            val_loss = F.binary_cross_entropy(val_pred, Y_val, weight=w_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and epoch >= 30:
            break

        if epoch % 20 == 0:
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        policy.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss, "epochs_trained": epoch, "history": history}


# ---------------------------------------------------------------------------
# Training: REINFORCE with baseline
# ---------------------------------------------------------------------------


def train_epoch(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    traces: list[ScoreTrace],
    *,
    window: int = 4,
    efficiency_alpha: float = 0.0,
    fp_penalty: float = 2.0,
    batch_size: int = 64,
    clip_grad: float = 1.0,
    entropy_bonus: float = 0.05,
) -> dict[str, float]:
    """REINFORCE with per-class baseline and entropy regularization.

    Uses separate baselines for erroneous and all-correct episodes to reduce
    variance and prevent collapse to always-stop degenerate solution.
    """
    policy.train()
    random.shuffle(traces)

    total_loss = 0.0
    total_reward = 0.0
    n_batches = 0

    for batch_start in range(0, len(traces), batch_size):
        batch = traces[batch_start : batch_start + batch_size]
        erroneous_returns: list[float] = []
        correct_returns: list[float] = []
        erroneous_log_probs: list[list[torch.Tensor]] = []
        correct_log_probs: list[list[torch.Tensor]] = []
        all_stop_probs: list[torch.Tensor] = []  # for entropy bonus

        for trace in batch:
            log_probs_ep, stopped_at = rollout_episode(
                trace, policy, window=window, deterministic=False, training=True
            )
            if not log_probs_ep:
                continue
            reward = compute_reward(
                trace,
                stopped_at=stopped_at,
                efficiency_alpha=efficiency_alpha,
                fp_penalty=fp_penalty,
            )
            if trace.is_all_correct:
                correct_returns.append(reward)
                correct_log_probs.append(log_probs_ep)
            else:
                erroneous_returns.append(reward)
                erroneous_log_probs.append(log_probs_ep)

        # Per-class baselines
        baseline_err = statistics.mean(erroneous_returns) if erroneous_returns else 0.0
        baseline_cor = statistics.mean(correct_returns) if correct_returns else 0.0

        losses = []
        all_returns = erroneous_returns + correct_returns
        all_log_probs = erroneous_log_probs + correct_log_probs
        all_baselines = [baseline_err] * len(erroneous_returns) + [baseline_cor] * len(correct_returns)

        for ret, lp_ep, bsl in zip(all_returns, all_log_probs, all_baselines):
            advantage = ret - bsl
            lp_stack = torch.stack(lp_ep)
            pg_loss = -advantage * lp_stack.sum()
            # Entropy bonus: -sum(p * log_p) = sum(lp * exp(lp)) (positive for uniform)
            # Approximate: just penalize low-entropy (high-confidence) by -entropy_bonus * log_p
            entropy_approx = -lp_stack.mean()  # higher = more entropy (less certain)
            ep_loss = pg_loss - entropy_bonus * entropy_approx
            losses.append(ep_loss)

        if not losses:
            continue

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        total_reward += statistics.mean(all_returns)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_reward": total_reward / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    binary_f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    mean_steps_frac: float
    mean_reward: float


def evaluate_policy(
    policy: nn.Module,
    traces: list[ScoreTrace],
    *,
    window: int = 4,
    efficiency_alpha: float = 0.0,
    fp_penalty: float = 2.0,
) -> EvalResult:
    policy.eval()
    tp = fp = fn = tn = 0
    steps_fracs = []
    rewards = []

    for trace in traces:
        _, stopped_at = rollout_episode(
            trace, policy, window=window, deterministic=True, training=False
        )
        predicted_erroneous = stopped_at is not None
        reward = compute_reward(
            trace, stopped_at=stopped_at, efficiency_alpha=efficiency_alpha, fp_penalty=fp_penalty
        )
        rewards.append(reward)

        steps_used = (stopped_at + 1) if stopped_at is not None else trace.n_steps
        steps_fracs.append(steps_used / trace.n_steps if trace.n_steps > 0 else 1.0)

        # Binary correct: predicted_erroneous AND stopped at or before first-bad
        if trace.is_all_correct:
            if predicted_erroneous:
                fp += 1
            else:
                tn += 1
        else:
            # correct = stopped_at is not None AND stopped_at <= trace.label
            if predicted_erroneous and stopped_at <= trace.label:
                tp += 1
            elif predicted_erroneous:
                # flagged but too late → counts as FP for erroneous? or FN?
                # Per ABR-lite logic: stopped_at > label is not a correct decision
                # We treat it as FN (missed the error in time)
                fn += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return EvalResult(
        binary_f1=f1,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        mean_steps_frac=statistics.mean(steps_fracs),
        mean_reward=statistics.mean(rewards),
    )


def evaluate_heuristic_threshold(
    traces: list[ScoreTrace],
    *,
    tau: float,
) -> EvalResult:
    """Evaluate the best heuristic baseline (threshold_only at given tau)."""
    tp = fp = fn = tn = 0
    steps_fracs = []
    rewards = []

    for trace in traces:
        stopped_at = None
        for t, score in enumerate(trace.scores):
            if score < tau:
                stopped_at = t
                break
        predicted_erroneous = stopped_at is not None
        reward = compute_reward(trace, stopped_at=stopped_at)
        rewards.append(reward)

        steps_used = (stopped_at + 1) if stopped_at is not None else trace.n_steps
        steps_fracs.append(steps_used / trace.n_steps if trace.n_steps > 0 else 1.0)

        if trace.is_all_correct:
            if predicted_erroneous:
                fp += 1
            else:
                tn += 1
        else:
            # Binary detection: ANY stop on erroneous = TP (matches F2 simulation metric)
            if predicted_erroneous:
                tp += 1
            else:
                fn += 1

    # Also compute balanced_f1 = harmonic mean of acc_err and acc_cor (same as F2 sim)
    acc_err = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc_cor = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_f1 = (2 * acc_err * acc_cor / (acc_err + acc_cor)) if (acc_err + acc_cor) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = acc_err
    positive_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return EvalResult(
        binary_f1=balanced_f1,  # report balanced_f1 (= F2 simulation primary metric)
        precision=acc_cor,       # repurpose: acc_correct (specificity)
        recall=acc_err,          # recall = acc_erroneous (sensitivity)
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        mean_steps_frac=statistics.mean(steps_fracs),
        mean_reward=statistics.mean(rewards),
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def run_heuristic_tau_sweep(
    traces: list[ScoreTrace],
    *,
    taus: list[float] | None = None,
) -> list[dict[str, Any]]:
    if taus is None:
        taus = [round(0.05 * i, 2) for i in range(1, 20)]
    results = []
    for tau in taus:
        r = evaluate_heuristic_threshold(traces, tau=tau)
        results.append(
            {
                "tau": tau,
                "binary_f1": r.binary_f1,
                "precision": r.precision,
                "recall": r.recall,
                "mean_steps_frac": r.mean_steps_frac,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "tn": r.tn,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main training + eval pipeline
# ---------------------------------------------------------------------------


def build_policy(
    arch: str,
    *,
    hidden_dim: int = 64,
    window: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    if arch == "linear":
        return LinearPolicy(feature_dim(window))
    elif arch == "mlp":
        return MLPPolicy(feature_dim(window), hidden_dim=hidden_dim, dropout=dropout)
    elif arch == "gru":
        return GRUPolicy(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored-rows",
        type=Path,
        required=True,
        help="Path to scored_rows.jsonl from phase_e_eval_benchmark.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write results and checkpoints",
    )
    parser.add_argument("--arch", choices=["linear", "mlp", "gru"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=4, help="Score history window for linear/mlp")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--efficiency-alpha", type=float, default=0.0, help="Efficiency bonus weight")
    parser.add_argument("--fp-penalty", type=float, default=2.0, help="False-alarm penalty multiplier (>1 breaks always-stop local optimum)")
    parser.add_argument("--entropy-bonus", type=float, default=0.05, help="Entropy regularization weight")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs without val improvement)",
    )
    parser.add_argument(
        "--mode",
        choices=["reinforce", "supervised", "reinforce_bc_warmstart"],
        default="supervised",
        help=(
            "Training mode: "
            "'supervised' = BCE on per-step oracle labels (most stable); "
            "'reinforce' = REINFORCE policy gradient; "
            "'reinforce_bc_warmstart' = BC pre-train then REINFORCE fine-tune"
        ),
    )
    parser.add_argument("--oracle-tau", type=float, default=0.38, help="Threshold for BC oracle (used in bc_warmstart mode)")
    parser.add_argument("--bc-epochs", type=int, default=30, help="BC pre-training epochs (reinforce_bc_warmstart mode)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load and split data
    print(f"Loading traces from {args.scored_rows}")
    traces = load_traces(args.scored_rows)
    train_traces, val_traces, test_traces = split_traces(
        traces, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )
    print(
        f"Split: train={len(train_traces)}, val={len(val_traces)}, test={len(test_traces)}"
    )

    # Heuristic baselines on test set
    print("\n=== Heuristic threshold sweep (val set) ===")
    tau_sweep = run_heuristic_tau_sweep(val_traces)
    best_heuristic = max(tau_sweep, key=lambda r: r["binary_f1"])
    print(
        f"  Best heuristic: tau={best_heuristic['tau']:.2f} "
        f"F1={best_heuristic['binary_f1']:.4f} "
        f"steps_frac={best_heuristic['mean_steps_frac']:.3f}"
    )

    # Save tau sweep
    (args.output_dir / "tau_sweep_val.json").write_text(
        json.dumps(tau_sweep, indent=2), encoding="utf-8"
    )

    # Build policy
    policy = build_policy(args.arch, hidden_dim=args.hidden_dim, window=args.window)
    n_params = sum(p.numel() for p in policy.parameters())

    # GRU is a sequential rollout policy — supervised/bc modes call policy(X) batch API
    # which GRUPolicy doesn't support. Auto-downgrade to reinforce.
    effective_mode = args.mode
    if isinstance(policy, GRUPolicy) and args.mode != "reinforce":
        print(f"  [NOTE] GRUPolicy: mode '{args.mode}' requires batch forward() which GRU doesn't support.")
        print(f"  [NOTE] Auto-switching to 'reinforce' mode for GRU.")
        effective_mode = "reinforce"

    print(f"\n=== Training {args.arch} policy ({n_params} params) — mode: {effective_mode} ===")

    history: list[dict[str, Any]] = []
    best_val_f1 = 0.0
    best_state = None
    epoch = 0

    if effective_mode == "supervised":
        # ----------------------------------------------------------------
        # Supervised BCE on per-step sequence labels
        # ----------------------------------------------------------------
        sup_result = train_supervised(
            policy,
            train_traces,
            val_traces,
            window=args.window,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size * 4,
            patience=args.patience,
        )
        epoch = sup_result["epochs_trained"]
        history = sup_result.get("history", [])
        print(f"  Supervised training done: epochs={epoch}, best_val_loss={sup_result['best_val_loss']:.4f}")
        best_val_f1 = evaluate_policy(
            policy, val_traces, window=args.window, fp_penalty=args.fp_penalty
        ).binary_f1
        best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    else:
        # ----------------------------------------------------------------
        # REINFORCE (optionally with BC warmstart)
        # ----------------------------------------------------------------
        if effective_mode == "reinforce_bc_warmstart":
            print(f"  BC pre-training for {args.bc_epochs} epochs (oracle_tau={args.oracle_tau})...")
            pretrain_bc(
                policy,
                train_traces,
                oracle_tau=args.oracle_tau,
                window=args.window,
                lr=args.lr * 3,
                epochs=args.bc_epochs,
            )
            # Evaluate after BC warmstart
            bc_val = evaluate_policy(policy, val_traces, window=args.window, fp_penalty=args.fp_penalty)
            print(f"  After BC warmstart: val_f1={bc_val.binary_f1:.4f} steps={bc_val.mean_steps_frac:.3f}")

        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                policy,
                optimizer,
                train_traces,
                window=args.window,
                efficiency_alpha=args.efficiency_alpha,
                fp_penalty=args.fp_penalty,
                batch_size=args.batch_size,
                clip_grad=args.clip_grad,
                entropy_bonus=args.entropy_bonus,
            )
            scheduler.step()

            if epoch % 10 == 0 or epoch <= 5:
                val_result = evaluate_policy(
                    policy, val_traces, window=args.window,
                    efficiency_alpha=args.efficiency_alpha, fp_penalty=args.fp_penalty
                )
                entry = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_mean_reward": train_metrics["mean_reward"],
                    "val_f1": val_result.binary_f1,
                    "val_steps_frac": val_result.mean_steps_frac,
                }
                history.append(entry)
                print(
                    f"  epoch={epoch:3d} "
                    f"loss={train_metrics['loss']:.4f} "
                    f"reward={train_metrics['mean_reward']:+.3f} "
                    f"val_f1={val_result.binary_f1:.4f} "
                    f"steps={val_result.mean_steps_frac:.3f}"
                )

                if val_result.binary_f1 > best_val_f1:
                    best_val_f1 = val_result.binary_f1
                    best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience and epoch >= 50:
                    print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    # Restore best
    if best_state is not None:
        policy.load_state_dict(best_state)

    # Final evaluation on test set
    test_result = evaluate_policy(
        policy, test_traces, window=args.window,
        efficiency_alpha=args.efficiency_alpha, fp_penalty=args.fp_penalty
    )
    heuristic_test = evaluate_heuristic_threshold(test_traces, tau=best_heuristic["tau"])

    print(f"\n=== Final test results ===")
    print(
        f"  RL policy   ({args.arch}): "
        f"F1={test_result.binary_f1:.4f} "
        f"P={test_result.precision:.4f} "
        f"R={test_result.recall:.4f} "
        f"steps={test_result.mean_steps_frac:.3f} "
        f"[TP={test_result.tp} FP={test_result.fp} FN={test_result.fn} TN={test_result.tn}]"
    )
    print(
        f"  Heuristic (tau={best_heuristic['tau']:.2f}): "
        f"F1={heuristic_test.binary_f1:.4f} "
        f"P={heuristic_test.precision:.4f} "
        f"R={heuristic_test.recall:.4f} "
        f"steps={heuristic_test.mean_steps_frac:.3f}"
    )

    delta_f1 = test_result.binary_f1 - heuristic_test.binary_f1
    print(f"\n  ΔF1 (RL - heuristic) = {delta_f1:+.4f}")
    if delta_f1 > 0.005:
        verdict = "RL WINS"
    elif delta_f1 < -0.005:
        verdict = "HEURISTIC WINS"
    else:
        verdict = "ROUGHLY EQUAL"
    print(f"  Verdict: {verdict}")

    # Save results
    result = {
        "run_name": f"rl_controller_{args.arch}_{run_ts}",
        "arch": args.arch,
        "hidden_dim": args.hidden_dim,
        "window": args.window,
        "lr": args.lr,
        "epochs_trained": epoch,
        "efficiency_alpha": args.efficiency_alpha,
        "seed": args.seed,
        "n_train": len(train_traces),
        "n_val": len(val_traces),
        "n_test": len(test_traces),
        "rl_policy": {
            "binary_f1": test_result.binary_f1,
            "precision": test_result.precision,
            "recall": test_result.recall,
            "mean_steps_frac": test_result.mean_steps_frac,
            "tp": test_result.tp,
            "fp": test_result.fp,
            "fn": test_result.fn,
            "tn": test_result.tn,
        },
        "heuristic_baseline": {
            "tau": best_heuristic["tau"],
            "binary_f1": heuristic_test.binary_f1,
            "precision": heuristic_test.precision,
            "recall": heuristic_test.recall,
            "mean_steps_frac": heuristic_test.mean_steps_frac,
        },
        "delta_f1": delta_f1,
        "verdict": verdict,
        "best_val_f1": best_val_f1,
        "training_history": history,
    }

    out_file = args.output_dir / f"results_{args.arch}_{run_ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")

    # Save checkpoint
    ckpt_file = args.output_dir / f"policy_{args.arch}_{run_ts}.pt"
    torch.save(
        {
            "arch": args.arch,
            "hidden_dim": args.hidden_dim,
            "window": args.window,
            "state_dict": policy.state_dict(),
            "test_f1": test_result.binary_f1,
        },
        ckpt_file,
    )
    print(f"Saved checkpoint to {ckpt_file}")


if __name__ == "__main__":
    main()

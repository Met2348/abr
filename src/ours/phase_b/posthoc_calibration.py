"""Post-hoc calibration helpers for Phase C value-head diagnostics.

Why this file exists
--------------------
The C2 value head can produce useful ranking signals while still being poorly
calibrated in absolute probability space. Post-hoc calibration lets us test
whether a lightweight mapping improves Brier/ECE without retraining the head.

Keeping this logic in one module prevents train/eval script divergence and makes
the calibration method explicit in manifests.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TemperatureCalibrationConfig:
    """Configuration for scalar temperature scaling."""

    lr: float = 0.05
    max_iters: int = 200
    min_temperature: float = 0.05
    max_temperature: float = 10.0
    init_temperature: float = 1.0

    def validate(self) -> None:
        """Validate bounds before optimization starts."""
        if self.lr <= 0.0:
            raise ValueError("Temperature calibration `lr` must be > 0")
        if self.max_iters <= 0:
            raise ValueError("Temperature calibration `max_iters` must be > 0")
        if self.min_temperature <= 0.0:
            raise ValueError("`min_temperature` must be > 0")
        if self.max_temperature <= self.min_temperature:
            raise ValueError("`max_temperature` must be > min_temperature")
        if not (self.min_temperature <= self.init_temperature <= self.max_temperature):
            raise ValueError("`init_temperature` must lie within [min_temperature, max_temperature]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class IsotonicCalibrationConfig:
    """Configuration for isotonic post-hoc calibration.

    `min_points` prevents fitting degenerate tiny datasets silently.
    """

    min_points: int = 32

    def validate(self) -> None:
        """Validate isotonic-calibration settings."""
        if self.min_points <= 0:
            raise ValueError("Isotonic calibration `min_points` must be > 0")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""
        self.validate()
        return asdict(self)


def fit_temperature_scaler(
    *,
    logits: Any,
    targets: Any,
    torch_module: Any,
    config: TemperatureCalibrationConfig,
) -> dict[str, Any]:
    """Fit one scalar temperature for logits using BCE objective.

    This uses bounded optimization with a sigmoid parameterization so the fitted
    temperature cannot silently become non-positive or explode to invalid values.
    """
    config.validate()
    if logits.shape != targets.shape:
        raise ValueError(
            f"Temperature scaling expects equal shapes, got {tuple(logits.shape)!r} and {tuple(targets.shape)!r}"
        )
    if logits.numel() == 0:
        raise ValueError("Temperature scaling requires non-empty logits/targets")

    logits_detached = logits.detach()
    targets_detached = targets.detach()

    # 用 sigmoid 参数化温度，强制温度始终落在 [min_t, max_t]。
    min_t = float(config.min_temperature)
    max_t = float(config.max_temperature)
    range_t = max_t - min_t
    init_alpha = (float(config.init_temperature) - min_t) / range_t
    # Clamp away from 0/1 before inverse sigmoid.
    init_alpha = min(max(init_alpha, 1e-6), 1.0 - 1e-6)
    init_raw = math.log(init_alpha / (1.0 - init_alpha))
    raw = torch_module.nn.Parameter(
        torch_module.tensor(
            init_raw,
            dtype=logits_detached.dtype,
            device=logits_detached.device,
        )
    )
    optimizer = torch_module.optim.Adam([raw], lr=float(config.lr))

    def current_temperature() -> Any:
        alpha = torch_module.sigmoid(raw)
        return min_t + range_t * alpha

    with torch_module.no_grad():
        start_temp = float(current_temperature().item())
        start_loss = float(
            torch_module.nn.functional.binary_cross_entropy_with_logits(
                logits_detached / current_temperature(),
                targets_detached,
            ).item()
        )

    final_loss = start_loss
    for _ in range(int(config.max_iters)):
        optimizer.zero_grad(set_to_none=True)
        temp = current_temperature()
        loss = torch_module.nn.functional.binary_cross_entropy_with_logits(
            logits_detached / temp,
            targets_detached,
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch_module.no_grad():
        fitted_temp = float(current_temperature().item())

    return {
        "method": "temperature",
        "temperature": float(fitted_temp),
        "objective": "bce_with_logits",
        "start_temperature": float(start_temp),
        "start_loss": float(start_loss),
        "final_loss": float(final_loss),
        "config": config.to_dict(),
    }


def fit_isotonic_calibrator(
    *,
    scores: Any,
    targets: Any,
    torch_module: Any,
    config: IsotonicCalibrationConfig,
) -> dict[str, Any]:
    """Fit an isotonic-regression calibrator on probability scores.

    This implementation uses pool-adjacent-violators (PAV) with unit weights.
    The learned mapping is piecewise-constant on sorted score bins.
    """
    config.validate()
    if scores.shape != targets.shape:
        raise ValueError(
            f"Isotonic calibration expects equal shapes, got {tuple(scores.shape)!r} and {tuple(targets.shape)!r}"
        )
    if scores.numel() < int(config.min_points):
        raise ValueError(
            f"Isotonic calibration requires at least {config.min_points} points; got {scores.numel()}"
        )

    score_list = [float(v) for v in scores.detach().reshape(-1).cpu().tolist()]
    target_list = [float(v) for v in targets.detach().reshape(-1).cpu().tolist()]

    # 先按分数排序，再用 PAV 强制单调，避免出现“分数更高反而校准后更低”的反直觉映射。
    pairs = sorted(zip(score_list, target_list, strict=True), key=lambda pair: pair[0])
    blocks: list[dict[str, float]] = []
    for score, target in pairs:
        blocks.append(
            {
                "sum_w": 1.0,
                "sum_y": float(target),
                "left_x": float(score),
                "right_x": float(score),
            }
        )
        while len(blocks) >= 2:
            right = blocks[-1]
            left = blocks[-2]
            left_avg = left["sum_y"] / left["sum_w"]
            right_avg = right["sum_y"] / right["sum_w"]
            if left_avg <= right_avg:
                break
            merged = {
                "sum_w": left["sum_w"] + right["sum_w"],
                "sum_y": left["sum_y"] + right["sum_y"],
                "left_x": left["left_x"],
                "right_x": right["right_x"],
            }
            blocks.pop()
            blocks.pop()
            blocks.append(merged)

    right_edges = [float(block["right_x"]) for block in blocks]
    fitted_values = [
        float(min(max(block["sum_y"] / block["sum_w"], 0.0), 1.0))
        for block in blocks
    ]

    calibrated = _apply_isotonic_mapping(
        raw_scores=score_list,
        right_edges=right_edges,
        fitted_values=fitted_values,
    )
    start_brier = _brier(score_list, target_list)
    final_brier = _brier(calibrated, target_list)
    return {
        "method": "isotonic",
        "objective": "brier",
        "start_brier": float(start_brier),
        "final_brier": float(final_brier),
        "num_bins": int(len(fitted_values)),
        "right_edges": right_edges,
        "fitted_values": fitted_values,
        "config": config.to_dict(),
    }


def apply_temperature_scaler(
    *,
    logits: Any,
    temperature: float,
    torch_module: Any,
) -> Any:
    """Apply one scalar temperature and return calibrated sigmoid scores."""
    if float(temperature) <= 0.0:
        raise ValueError("`temperature` must be > 0")
    return torch_module.sigmoid(logits / float(temperature))


def apply_isotonic_calibrator(
    *,
    scores: Any,
    calibrator: dict[str, Any],
    torch_module: Any,
) -> Any:
    """Apply an isotonic calibrator payload and return calibrated scores tensor."""
    right_edges = calibrator.get("right_edges")
    fitted_values = calibrator.get("fitted_values")
    if not isinstance(right_edges, list) or not isinstance(fitted_values, list):
        raise ValueError("Isotonic calibrator payload must contain list `right_edges` and `fitted_values`")
    if len(right_edges) != len(fitted_values) or not right_edges:
        raise ValueError("Isotonic calibrator payload has inconsistent bin arrays")
    raw = [float(v) for v in scores.detach().reshape(-1).cpu().tolist()]
    calibrated = _apply_isotonic_mapping(
        raw_scores=raw,
        right_edges=[float(v) for v in right_edges],
        fitted_values=[float(v) for v in fitted_values],
    )
    tensor = torch_module.tensor(calibrated, dtype=torch_module.float32, device=scores.device)
    return tensor.reshape(scores.shape)


def apply_posthoc_calibration(
    *,
    logits: Any | None,
    scores: Any | None,
    calibrator: dict[str, Any],
    torch_module: Any,
) -> Any:
    """Apply one persisted post-hoc calibrator payload to logits.

    Current supported methods:
    - `temperature` (requires logits)
    - `isotonic` (requires scores, or logits as fallback)
    """
    method = str(calibrator.get("method", "")).strip().lower()
    if method == "temperature":
        if logits is None:
            raise ValueError("Temperature calibration requires non-null `logits`")
        return apply_temperature_scaler(
            logits=logits,
            temperature=float(calibrator["temperature"]),
            torch_module=torch_module,
        )
    if method == "isotonic":
        score_tensor = scores
        if score_tensor is None:
            if logits is None:
                raise ValueError("Isotonic calibration requires `scores` or `logits`")
            score_tensor = torch_module.sigmoid(logits)
        return apply_isotonic_calibrator(
            scores=score_tensor,
            calibrator=calibrator,
            torch_module=torch_module,
        )
    raise ValueError(f"Unsupported post-hoc calibration method: {method!r}")


def _apply_isotonic_mapping(
    *,
    raw_scores: list[float],
    right_edges: list[float],
    fitted_values: list[float],
) -> list[float]:
    """Apply piecewise-constant isotonic mapping to raw scores."""
    calibrated: list[float] = []
    last_idx = len(right_edges) - 1
    for score in raw_scores:
        idx = bisect_right(right_edges, float(score))
        if idx > last_idx:
            idx = last_idx
        calibrated.append(float(fitted_values[idx]))
    return calibrated


def _brier(predictions: list[float], targets: list[float]) -> float:
    """Return mean squared error between probabilities and targets."""
    if len(predictions) != len(targets):
        raise ValueError("Brier helper expects aligned lists")
    if not predictions:
        raise ValueError("Brier helper expects non-empty lists")
    return float(sum((p - t) ** 2 for p, t in zip(predictions, targets, strict=True)) / len(predictions))

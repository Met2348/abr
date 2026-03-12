#!/usr/bin/env bash
# Phase F RL controller parameter sweep.
#
# 对 ABR-lite RL 控制器进行系统性参数扫描：
# - 架构: linear / mlp / gru
# - 隐藏层大小: 32 / 64 / 128
# - 学习率: 1e-3 / 3e-3 / 1e-2
# - efficiency_alpha: 0.0 / 0.1 / 0.2
# - 数据: pbr19 MATH / pbr19 GSM8K (if available)
#
# Usage:
#   bash scripts/run_phase_f_rl_sweep.sh
#   SWEEP_SUBSET=quick bash scripts/run_phase_f_rl_sweep.sh  # only key configs

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON_BIN:-python3}
SCORED_ROWS_MATH="assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_math_fulleval_20260311T123421Z/scored_rows.jsonl"
SCORED_ROWS_GSM="assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_gsm_fulleval_20260311T123421Z/scored_rows.jsonl"
OUT_ROOT="assets/artifacts/phase_f_rl_controller"
LOG_DIR="assets/artifacts/phase_e_logs/phase_f_rl"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

SWEEP_SUBSET=${SWEEP_SUBSET:-full}

run_config() {
    local name="$1"
    local scored_rows="$2"
    local arch="$3"
    local hidden="$4"
    local lr="$5"
    local alpha="$6"
    local seed="$7"
    local extra_args="${8:-}"
    local out_dir="$OUT_ROOT/${name}"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name (arch=$arch hidden=$hidden lr=$lr alpha=$alpha)"
    $PY scripts/phase_f_train_rl_controller.py \
        --scored-rows "$scored_rows" \
        --output-dir "$out_dir" \
        --arch "$arch" \
        --hidden-dim "$hidden" \
        --lr "$lr" \
        --efficiency-alpha "$alpha" \
        --seed "$seed" \
        --epochs 400 \
        --batch-size 64 \
        --patience 40 \
        $extra_args \
        > "$logfile" 2>&1
    # Print last few lines
    tail -6 "$logfile" | sed "s/^/  [$name] /"
}

# -----------------------------------------------------------------------
# MATH dataset sweep
# -----------------------------------------------------------------------
echo "==================================================================="
echo "Phase F RL Controller Sweep — dataset: ProcessBench MATH (PBR19)"
echo "==================================================================="

if [ "$SWEEP_SUBSET" = "quick" ]; then
    # Quick subset: just the most promising configs
    run_config "math_linear_s42"    "$SCORED_ROWS_MATH" linear 32  3e-3 0.0 42
    run_config "math_mlp64_s42"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0 42
    run_config "math_mlp128_s42"    "$SCORED_ROWS_MATH" mlp    128 1e-3 0.0 42
    run_config "math_gru32_s42"     "$SCORED_ROWS_MATH" gru    32  3e-3 0.0 42
    run_config "math_mlp64_eff01"   "$SCORED_ROWS_MATH" mlp    64  3e-3 0.1 42
else
    # Full sweep
    # Architecture ablation (mlp, fixed lr=3e-3, no efficiency bonus)
    run_config "math_linear_s42"      "$SCORED_ROWS_MATH" linear 32  3e-3 0.0 42
    run_config "math_mlp32_s42"       "$SCORED_ROWS_MATH" mlp    32  3e-3 0.0 42
    run_config "math_mlp64_s42"       "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0 42
    run_config "math_mlp128_s42"      "$SCORED_ROWS_MATH" mlp    128 3e-3 0.0 42
    run_config "math_gru32_s42"       "$SCORED_ROWS_MATH" gru    32  3e-3 0.0 42
    run_config "math_gru64_s42"       "$SCORED_ROWS_MATH" gru    64  3e-3 0.0 42

    # LR sweep (best arch = mlp64)
    run_config "math_mlp64_lr1e3"     "$SCORED_ROWS_MATH" mlp    64  1e-3 0.0 42
    run_config "math_mlp64_lr3e3"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0 42
    run_config "math_mlp64_lr1e2"     "$SCORED_ROWS_MATH" mlp    64  1e-2 0.0 42

    # Efficiency bonus sweep (mlp64, lr=3e-3)
    run_config "math_mlp64_eff00"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  42
    run_config "math_mlp64_eff01"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.1  42
    run_config "math_mlp64_eff02"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.2  42
    run_config "math_mlp64_eff03"     "$SCORED_ROWS_MATH" mlp    64  3e-3 0.3  42

    # Window ablation (mlp64, lr=3e-3)
    run_config "math_mlp64_win2"      "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  42 "--window 2"
    run_config "math_mlp64_win4"      "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  42 "--window 4"
    run_config "math_mlp64_win6"      "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  42 "--window 6"
    run_config "math_mlp64_win8"      "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  42 "--window 8"

    # Seed stability (best config)
    run_config "math_mlp64_s1"        "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  1
    run_config "math_mlp64_s7"        "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  7
    run_config "math_mlp64_s123"      "$SCORED_ROWS_MATH" mlp    64  3e-3 0.0  123

    # GRU with efficiency bonus
    run_config "math_gru32_eff01"     "$SCORED_ROWS_MATH" gru    32  3e-3 0.1  42
    run_config "math_gru64_eff01"     "$SCORED_ROWS_MATH" gru    64  3e-3 0.1  42
fi

# -----------------------------------------------------------------------
# GSM8K dataset sweep (if available)
# -----------------------------------------------------------------------
if [ -f "$SCORED_ROWS_GSM" ]; then
    echo ""
    echo "==================================================================="
    echo "Phase F RL Controller Sweep — dataset: ProcessBench GSM8K (PBR19)"
    echo "==================================================================="
    if [ "$SWEEP_SUBSET" = "quick" ]; then
        run_config "gsm_mlp64_s42"    "$SCORED_ROWS_GSM" mlp 64 3e-3 0.0 42
        run_config "gsm_gru32_s42"    "$SCORED_ROWS_GSM" gru 32 3e-3 0.0 42
    else
        run_config "gsm_linear_s42"   "$SCORED_ROWS_GSM" linear 32  3e-3 0.0 42
        run_config "gsm_mlp64_s42"    "$SCORED_ROWS_GSM" mlp    64  3e-3 0.0 42
        run_config "gsm_mlp128_s42"   "$SCORED_ROWS_GSM" mlp    128 1e-3 0.0 42
        run_config "gsm_gru32_s42"    "$SCORED_ROWS_GSM" gru    32  3e-3 0.0 42
        run_config "gsm_mlp64_eff01"  "$SCORED_ROWS_GSM" mlp    64  3e-3 0.1 42
        run_config "gsm_mlp64_eff02"  "$SCORED_ROWS_GSM" mlp    64  3e-3 0.2 42
    fi
fi

# -----------------------------------------------------------------------
# Collect results summary
# -----------------------------------------------------------------------
echo ""
echo "==================================================================="
echo "Results Summary"
echo "==================================================================="
$PY - <<'PYEOF'
import json, os, glob, sys
results = []
for f in sorted(glob.glob("assets/artifacts/phase_f_rl_controller/**/results_*.json", recursive=True)):
    try:
        r = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        results.append({
            "name": name,
            "arch": r.get("arch"),
            "f1_rl": r["rl_policy"]["binary_f1"],
            "f1_heur": r["heuristic_baseline"]["binary_f1"],
            "delta": r["delta_f1"],
            "steps_rl": r["rl_policy"]["mean_steps_frac"],
            "steps_heur": r["heuristic_baseline"]["mean_steps_frac"],
            "verdict": r["verdict"],
        })
    except Exception as e:
        print(f"  skip {f}: {e}")

results.sort(key=lambda r: -r["f1_rl"])
print(f"\n{'Name':<35} {'Arch':<8} {'RL F1':>7} {'Heur F1':>8} {'ΔF1':>7} {'RL steps':>9} {'Verdict'}")
print("-" * 95)
for r in results:
    print(f"{r['name']:<35} {r['arch']:<8} {r['f1_rl']:>7.4f} {r['f1_heur']:>8.4f} {r['delta']:>+7.4f} {r['steps_rl']:>9.3f} {r['verdict']}")

if results:
    best = results[0]
    print(f"\nBEST RL: {best['name']} F1={best['f1_rl']:.4f} ΔF1={best['delta']:+.4f} steps={best['steps_rl']:.3f}")
PYEOF

echo ""
echo "Sweep complete: $(date)"

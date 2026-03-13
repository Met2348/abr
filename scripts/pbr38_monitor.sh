#!/bin/bash
# PBR38 monitoring script - check experiment status
cd /home/zling/y/bcr/ref

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "=== PBR38A (LoRA+ctr) Log Tail ==="
tail -3 assets/artifacts/phase_e_logs/phase_e_pbr38a_train.log 2>/dev/null

echo ""
echo "=== PBR38B (consensus frozen) Log Tail ==="
tail -3 assets/artifacts/phase_e_logs/phase_e_pbr38b_train.log 2>/dev/null

echo ""
echo "=== PBR38E (implicit PRM) Log Tail ==="
tail -3 assets/artifacts/phase_e_logs/phase_e_pbr38e_implicit_prm.log 2>/dev/null

echo ""
echo "=== Completed runs (summary.json files) ==="
for d in assets/artifacts/phase_e_runs/phase_e_pbr38*/; do
    if [ -f "$d/summary.json" ]; then
        run=$(basename $d)
        math_f1=$(python3 -c "import json; d=json.load(open('$d/summary.json')); print(d.get('processbench_math_f1','N/A'))" 2>/dev/null)
        echo "  $run: MATH F1=$math_f1"
    fi
done
for d in assets/artifacts/phase_f_implicit_prm/pbr38e_*/; do
    if [ -f "$d/summary.json" ]; then
        run=$(basename $d)
        f1=$(python3 -c "import json; d=json.load(open('$d/summary.json')); print(d.get('processbench_math_f1','N/A'))" 2>/dev/null)
        echo "  $run: MATH F1=$f1"
    fi
done

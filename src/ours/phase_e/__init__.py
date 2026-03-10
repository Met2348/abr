"""Phase E: high-quality pair benchmark validation stack.

Why this package exists
-----------------------
Phase E changes the project center of gravity:

1. We no longer treat StrategyQA as the primary supervised benchmark for
   value-head learnability.
2. We first validate whether the ranking/value head is learnable on datasets
   that genuinely provide stronger process or pair supervision.
3. Only after that do we return to StrategyQA as a transfer target.

What this package contains
--------------------------
The package keeps the new benchmark-native logic separate from the older
Phase C / Phase D scripts so the repository does not keep accreting special
cases into one giant training file.

- `contracts.py`
  Phase E benchmark/source registry and naming contract.
- `pairs.py`
  Canonical pair-artifact preparation for Phase E sources.
- `runtime.py`
  Shared backbone loading, feature caching, and batched scoring helpers.
- `training.py`
  External-pair-only value-head training utilities.
- `benchmark_eval.py`
  ProcessBench / PRMBench-preview benchmark-native evaluators.
"""


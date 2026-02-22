"""
Quick dataset availability check for the BCR project.

Run:
  python check_gsm8k.py
"""

from datasets import load_dataset

TESTS = [
    ("gsm8k", "main", "train[:2]"),
    ("drop", None, "train[:2]"),
    ("lucasmccabe/logiqa", None, "train[:2]"),
    ("tasksource/strategy-qa", None, "train[:2]"),
    ("tasksource/proofwriter", "default", "train[:2]"),
    ("lukaemon/bbh", "boolean_expressions", "test[:2]"),
    ("EleutherAI/hendrycks_math", "algebra", "train[:2]"),
]


def check_one(repo_id: str, config: str | None, split: str) -> None:
    print(f"\n=== {repo_id} | config={config} | split={split} ===")
    try:
        if config is None:
            ds = load_dataset(repo_id, split=split)
        else:
            ds = load_dataset(repo_id, config, split=split)
        print(ds)
        print("first_sample_keys:", list(ds[0].keys()))
    except Exception as exc:
        print("FAILED:", type(exc).__name__, str(exc))


if __name__ == "__main__":
    for repo_id, config, split in TESTS:
        check_one(repo_id, config, split)

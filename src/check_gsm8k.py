"""Quick availability probe for the external dataset repos used by this project.

Why this file exists
--------------------
Before building a full loader pipeline, it is often useful to confirm that the raw
Hugging Face dataset repos are still reachable and still expose the expected fields.

What this file does
-------------------
- iterates through a small hard-coded list of dataset/config/split combinations,
- attempts to download a tiny slice from each repo,
- prints either the dataset summary and sample keys or a readable failure message.

Example
-------
```bash
python check_gsm8k.py
```
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
    """Probe one dataset repo/config/split combination and print the result.

    Example
    -------
    ```python
    check_one("gsm8k", "main", "train[:2]")
    ```
    """
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

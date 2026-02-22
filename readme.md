# BCR local setup notes

## Model files

Please manually put the ~15 GiB, 4 `*.safetensors` files into:

`assets/models/Qwen2.5-7B-Instruct`

Expected names:

- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

## Dataset downloads

Use:

```bash
bash download_datasets.sh
```

or with explicit target dir:

```bash
bash download_datasets.sh assets/datasets
```

The script includes fallbacks for datasets whose old IDs are invalid or disabled.

### Why some old commands fail

- `logiqa`, `strategyqa`, `proofwriter`, `bigbench_hard` without namespace:
  these IDs are not valid dataset repos on HF Hub.
- `hendrycks/competition_math`:
  this repo is currently disabled (403). Use `EleutherAI/hendrycks_math` instead.

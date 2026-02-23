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

## [Dev] How to recover the Codex dev context at a new chat?

- Locate the file named `chat_system_prompts.md` at the root, and Order Codex to read it so it may restore the requirements that it should strictly follow at each singe chat.

## [Dev] Where are the previous chats?

- The prompts since the very beginning of the project, can all be found at `chat_prompts.txt`. 

## [Dev] What do the two `TODO` files mean?

- The `TODO.md` assumes the BCR codes are released so that we may build upon their codes.
- However, since no codes are released, we decide to build our own codebase from scratch.
- The `TODO_ours.md` describes what we are going to do at each step, without the assistance of BCR.

## [Dev] What do the two `idea` files mean?
- The `idea_polish.md` contains ideas extracted from the original BCR idea and two following discussions.
- The `idea_formulation.md` is the file that explicitly expressed the ideas with math formula. The format is KaTeX friendly, so it can be properly rendered in Obsidian and Notion.
# Phase D Decision Update (2-Slide Deck Draft)

> 2026-03-10 update:
> this slide draft is now historical context for the "why external pairs" decision.
> The current stronger conclusion is:
> StrategyQA lacks PRM-grade step-quality supervision, so Phase D mainline
> validation should move to PRM-grade benchmarks first; StrategyQA remains a
> transfer/OOD target.

## Slide 1 - Why We Must Introduce External Pair Resources, and How

Title:
`Why External Pair Supervision Is Necessary in Phase D`

Problem snapshot:
1. Our D2/D3 runs show only marginal ranking gain, with AUC still near plateau.
2. PRM-as-clean-target helps coverage but does not fully fix pairwise ranking.
3. Pure 7B self-bootstrapped pairs are likely too noisy for primary supervision.

Why this is expected (from prior work):
1. Process supervision is data-noise sensitive and often requires high-density labels.
2. Process rewards can help, but only under careful pair construction/filtering.
3. Weak or ambiguous pairs commonly lead to stalled ranking metrics.

Decision:
1. Use external curated pair/step resources as the main pair signal source.
2. Keep 7B-generated pairs as late-stage augmentation only.

Execution plan:
1. Build mixed pair pool from curated resources + converted step labels.
2. Normalize to our C1/C2 schema with source tags and confidence fields.
3. Filter with teacher margins and ambiguity controls.
4. Train with source-balanced sampling and ranking-first objectives.
5. Promote only if ranking + robustness gates are both satisfied.

Success criteria:
1. Pair margin distribution improves and does not collapse.
2. `corr_pair_acc` and `corr_auc` improve together under equal budget.
3. Gains survive anti-gaming checks (length/style/BoN sensitivity).


## Slide 2 - Available External Resources for Pair Supervision

Title:
`Available Pair/PRM Resources We Can Use Now`

Task-specific reality check:
1. GSM8K: has usable third-party pair/preference datasets.
2. StrategyQA: no mainstream official chosen/rejected pair corpus.
3. Therefore: external resources are mandatory, especially for GSM8K bootstrapping;
   StrategyQA still needs in-project corruption + teacher scoring.

Direct pair resources (high priority):
1. `R-PRM` (chosen/rejected preference pairs)
   - https://huggingface.co/datasets/kevinpro/R-PRM
   - https://github.com/NJUNLP/R-PRM
2. `PRMBench_Preview` (original vs modified process traces)
   - https://huggingface.co/datasets/hitsmy/PRMBench_Preview
3. Community GSM8K pair datasets (optional):
   - https://huggingface.co/datasets/Rudra-ai/ai-responses-gsm8k-405b-dpo
   - https://huggingface.co/datasets/Rudra-ai/ai-responses-gsm8k-70b-update-dpo
   - https://huggingface.co/datasets/Genesis-AI-Labs/GAIL-gsm8k-preference-small

Step-level resources (convert to pairs):
1. `PRM800K` (step ratings, OpenAI)
   - https://github.com/openai/prm800k
2. `Math-Shepherd` (step-level +/- labels)
   - https://huggingface.co/datasets/peiyi9979/Math-Shepherd
3. `RLHFlow PRM Data` (Mistral/Deepseek process labels)
   - https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data
   - https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data
   - https://github.com/RLHFlow/RLHF-Reward-Modeling

Evaluation/diagnostic benchmarks:
1. `ProcessBench`
   - https://arxiv.org/abs/2412.06559
   - https://github.com/QwenLM/ProcessBench
2. `PRMBench`
   - https://arxiv.org/abs/2504.16828
   - https://github.com/ssmisya/PRMBench

Recommended adoption order:
1. Start with direct pair resources.
2. Add converted step-level pairs with strict filtering.
3. Use benchmarks for validation, not as the only training source.

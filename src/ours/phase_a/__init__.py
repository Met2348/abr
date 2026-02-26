"""Phase A baseline utilities.

Phase A scope:
- prompt construction
- deterministic split
- inference-time answer extraction
- evaluation utilities
"""

from .answer_extraction import ExtractedAnswer, answers_equivalent, extract_answer, normalize_gold_answer
from .contracts import (
    PredictionRecord,
    PreparedSample,
    PromptTemplateSpec,
    ScoredPrediction,
    TargetStyle,
)
from .evaluator import EvalSummary, evaluate_predictions, score_prediction
from .prompt_builder import (
    PROMPT_TEMPLATE_REGISTRY,
    build_prepared_sample,
    list_template_versions,
    resolve_template,
)
from .splitting import SplitConfig, assign_split, split_ids

__all__ = [
    "TargetStyle",
    "PromptTemplateSpec",
    "PreparedSample",
    "PredictionRecord",
    "ScoredPrediction",
    "SplitConfig",
    "assign_split",
    "split_ids",
    "ExtractedAnswer",
    "extract_answer",
    "normalize_gold_answer",
    "answers_equivalent",
    "EvalSummary",
    "score_prediction",
    "evaluate_predictions",
    "PROMPT_TEMPLATE_REGISTRY",
    "list_template_versions",
    "resolve_template",
    "build_prepared_sample",
]

"""不同 padding 方向下的 last-token 池化回归测试。

Regression tests for last-token pooling under different padding sides.
"""

from __future__ import annotations

import torch

from ours.phase_b.value_head import pool_last_token


def test_pool_last_token_handles_right_padding() -> None:
    hidden_states = torch.arange(10, dtype=torch.float32).view(2, 5, 1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    pooled = pool_last_token(hidden_states, attention_mask, torch_module=torch)

    assert pooled.squeeze(-1).tolist() == [2.0, 9.0]


def test_pool_last_token_handles_left_padding() -> None:
    hidden_states = torch.arange(10, dtype=torch.float32).view(2, 5, 1)
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    pooled = pool_last_token(hidden_states, attention_mask, torch_module=torch)

    assert pooled.squeeze(-1).tolist() == [4.0, 9.0]

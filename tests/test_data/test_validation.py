from __future__ import annotations

import pytest
import torch

from src.data.validation import validate_input_batch


def test_validate_accepts_valid_batch() -> None:
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    y = torch.tensor([1, 2], dtype=torch.long)
    validate_input_batch(x, y)


def test_validate_rejects_out_of_range() -> None:
    x = torch.full((2, 3, 32, 32), 1.1, dtype=torch.float32)
    y = torch.tensor([1, 2], dtype=torch.long)
    with pytest.raises(AssertionError, match="\\[0, 1\\]"):
        validate_input_batch(x, y)


def test_validate_rejects_wrong_dtype() -> None:
    x = torch.rand(2, 3, 32, 32, dtype=torch.float64)
    y = torch.tensor([1, 2], dtype=torch.long)
    with pytest.raises(AssertionError, match="float32"):
        validate_input_batch(x, y)


def test_validate_rejects_shape_mismatch() -> None:
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.long)
    with pytest.raises(AssertionError, match="Batch size mismatch"):
        validate_input_batch(x, y)

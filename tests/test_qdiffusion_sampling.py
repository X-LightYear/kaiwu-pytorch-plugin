"""Tests for the standalone QDiffusion sampling helpers."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_top_k_top_p_filtering_does_not_mutate_input_logits():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    original = logits.clone()

    filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

    assert torch.equal(logits, original)
    assert torch.isneginf(filtered[0, 2:]).all()
    assert torch.equal(filtered[0, :2], original[0, :2])
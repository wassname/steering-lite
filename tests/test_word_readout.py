"""Tests for the newer prompt-free j-steer word readout.

Implementation: Codex.
"""

from types import SimpleNamespace

import torch
from torch import nn

import steering_lite as sl


class TinyTokenizer:
    words = ("red", "blue", "green", "yellow", "1", "<|end|>")

    def __len__(self):
        return len(self.words)

    def decode(self, token_ids, **_kwargs):
        return self.words[token_ids[0]]


def tiny_model():
    model = nn.Module()
    model.lm_head = nn.Linear(3, len(TinyTokenizer.words), bias=False)
    model.model = SimpleNamespace(norm=nn.Identity())
    model.lm_head.weight.data = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0],
        ]
    )
    return model


def test_readout_words_decodes_both_residual_poles() -> None:
    vector = sl.Vector(
        sl.MeanDiffC(layers=(0,)),
        {0: {}},
        {0: {"v": torch.tensor([[1.0, 0.0, 0.0]])}},
    )
    readout = sl.readout_words(tiny_model(), TinyTokenizer(), vector, k=1)
    assert readout["layers"]["0.v"]["pos"] == ["red"]
    assert readout["layers"]["0.v"]["neg"] == ["blue"]


def test_readout_words_reconstructs_super_sspace_direction() -> None:
    vector = sl.Vector(
        sl.SuperSSpaceC(layers=(0,)),
        {
            0: {
                "U_r": torch.tensor(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 0.0],
                    ]
                ),
                "sqrtS": torch.tensor([2.0, 1.0]),
            }
        },
        {0: {"dS": torch.tensor([[0.0, 1.0]])}},
    )
    readout = sl.readout_words(tiny_model(), TinyTokenizer(), vector, k=1)
    assert readout["layers"]["0.dS"]["pos"] == ["red"]
    assert readout["layers"]["0.dS"]["neg"] == ["blue"]

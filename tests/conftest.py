"""Shared fixtures: a tiny word-level tokenizer and a tiny random ModernBERT DecisionModel."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest
import torch
from laya.common import DecisionModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

from tollgate.train import train as tt

TINY_MAX_LEN = 256
TINY_HEAD_MAX_LEN = 96

TinyModelFactory = Callable[[], tuple[DecisionModel, dict[str, Any]]]


@pytest.fixture(scope="session")
def tok() -> PreTrainedTokenizerFast:
    text = json.dumps(tt.QUESTIONS) + " ".join(tt._SYNTHETIC_WORDS) + " synthetic smoke row query"
    words = sorted(set(re.findall(r"\w+|[^\w\s]", text)))
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture(scope="session")
def tiny_model(tok: PreTrainedTokenizerFast) -> TinyModelFactory:
    """Factory for a fresh, seeded tiny DecisionModel plus a matching laya base config."""

    def make() -> tuple[DecisionModel, dict[str, Any]]:
        torch.manual_seed(0)
        ecfg = ModernBertConfig(
            vocab_size=len(tok),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=512,
            global_attn_every_n_layers=1,
            local_attention=64,
            pad_token_id=tok.pad_token_id,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            bos_token_id=tok.cls_token_id,
            eos_token_id=tok.sep_token_id,
        )
        encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        base_cfg = {
            "encoder": "tiny-modernbert",
            "head_layers": 1,
            "max_len": TINY_MAX_LEN,
            "head_max_len": TINY_HEAD_MAX_LEN,
            "act_costs": {"escalate": 0.5},
            "temperature": [1.6, 1.2, 1.9],
            "temperature_by_options": {"noul:2": 1.9},
        }
        return DecisionModel(encoder, head_layers=1, n_act=2), base_cfg

    return make

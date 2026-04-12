"""Tests for evaluation utilities."""

import pytest

from kiv.eval_utils import (
    FILLER_PARAGRAPHS,
    NeedlePrompt,
    _encode_text,
    _repeat_to_length,
)


class FakeTokenizer:
    """Minimal tokenizer for testing — 1 char = 1 token."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Minimal chat template: just wrap content
        content = messages[0]["content"]
        return f"<s>user\n{content}\nmodel\n"


def test_filler_paragraphs_exist():
    assert len(FILLER_PARAGRAPHS) == 10
    for p in FILLER_PARAGRAPHS:
        assert len(p) > 100


def test_encode_text():
    tok = FakeTokenizer()
    ids = _encode_text(tok, "hello")
    assert ids == [104, 101, 108, 108, 111]


def test_repeat_to_length():
    assert _repeat_to_length([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]
    assert _repeat_to_length([1, 2], 2) == [1, 2]
    assert _repeat_to_length([1], 5) == [1, 1, 1, 1, 1]
    assert _repeat_to_length([1, 2, 3], 0) == []


def test_repeat_to_length_rejects_empty():
    with pytest.raises(ValueError):
        _repeat_to_length([], 5)


def test_needle_prompt_dataclass():
    import torch
    p = NeedlePrompt(
        input_ids=torch.zeros(1, 10, dtype=torch.long),
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        context_length=10,
        filler_budget=5,
        needle_start=2,
        question_start=7,
        actual_depth=0.4,
    )
    assert p.context_length == 10
    assert p.actual_depth == 0.4

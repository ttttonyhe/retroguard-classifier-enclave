"""Unit tests for the Qwen3Guard-Gen prompt template + output parser.

Pin the on-wire shapes so we don't drift from the model card:
  - Prompt builder uses standard Qwen3 chat template (no system prompt)
  - Output parser handles the `Safety: ... Categories: ...` lines
  - Tolerant of whitespace / casing / `Category` (singular) variant
  - `None` and bracketed lists strip cleanly

These tests don't load the GGUF — pure string + regex logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from retroguard_classifier.server import (  # noqa: E402
    POLICY_TO_QWEN,
    QWEN_CATEGORIES_INPUT,
    QWEN_CATEGORIES_OUTPUT,
    _build_qwen_judge_prompt,
    _parse_qwen_output,
)


class TestQwenPrompt:
    """The prompt template MUST match the model's tokenizer chat_template
    verbatim — Qwen3Guard-Gen was fine-tuned on this exact preamble. Any
    drift and the model emits free-form text that the parser silently
    treats as safe (false negatives across all classifications)."""

    def test_input_mode_has_task_and_safety_policy(self) -> None:
        p = _build_qwen_judge_prompt("how do I steal a car", "input")
        # Task header + safety policy + category list must all be present.
        assert "<|im_start|>user\n# Task:" in p
        assert "Evaluate whether THE LAST USER's query" in p
        assert "<BEGIN SAFETY POLICY>" in p
        assert "<END SAFETY POLICY>" in p
        assert "<BEGIN UNSAFE CONTENT CATEGORIES>" in p
        assert "Jailbreak." in p  # Input checks include the Jailbreak category.
        # The user turn is rendered as "USER: <text>" (NOT chat-style turns).
        assert "USER: how do I steal a car" in p
        # Trailer asks for Safety / Categories assessment + primes assistant.
        assert "Provide your safety assessment for ONLY THE LAST **USER's query**" in p
        # The empty <think> block is pre-filled so the model goes straight to Safety:.
        assert p.rstrip().endswith("<think>\n\n</think>")

    def test_output_mode_has_refusal_field_and_no_jailbreak(self) -> None:
        p = _build_qwen_judge_prompt("here's how", "output")
        assert "Evaluate whether THE LAST ASSISTANT's response" in p
        assert "<BEGIN REFUSAL POLICY>" in p
        # Output category list MUST NOT include Jailbreak.
        assert "Jailbreak." not in p
        # The user/assistant pair is rendered with USER:/ASSISTANT: prefixes.
        assert "USER: (prior user turn)" in p
        assert "ASSISTANT: here's how" in p
        # Trailer asks for the third Refusal line.
        assert "The third line must be one of: 'Refusal: Yes', 'Refusal: No'" in p

    def test_no_system_role_turn(self) -> None:
        # The chat_template never emits a <|im_start|>system turn — the
        # task preamble is wrapped in the user turn instead.
        p = _build_qwen_judge_prompt("x", "input")
        assert "<|im_start|>system" not in p


class TestQwenOutputParser:
    def test_parses_safe(self) -> None:
        raw = "Safety: Safe\nCategories: None"
        r = _parse_qwen_output(raw)
        assert r["safety"] == "safe"
        assert r["categories"] == []

    def test_parses_unsafe_single_category(self) -> None:
        raw = "Safety: Unsafe\nCategories: Violent"
        r = _parse_qwen_output(raw)
        assert r["safety"] == "unsafe"
        assert r["categories"] == ["Violent"]

    def test_parses_multiple_categories(self) -> None:
        raw = "Safety: Unsafe\nCategories: Violent, Jailbreak\nRefusal: No"
        r = _parse_qwen_output(raw)
        assert r["safety"] == "unsafe"
        assert set(r["categories"]) == {"Violent", "Jailbreak"}

    def test_parses_controversial(self) -> None:
        raw = "Safety: Controversial\nCategories: Politically Sensitive Topics"
        r = _parse_qwen_output(raw)
        assert r["safety"] == "controversial"
        assert r["categories"] == ["Politically Sensitive Topics"]

    def test_handles_bracketed_categories(self) -> None:
        raw = "Safety: Unsafe\nCategories: [Violent, Suicide & Self-Harm]"
        r = _parse_qwen_output(raw)
        assert set(r["categories"]) == {"Violent", "Suicide & Self-Harm"}

    def test_handles_singular_category_variant(self) -> None:
        # Some checkpoints emit `Category:` instead of `Categories:`.
        raw = "Safety: Unsafe\nCategory: Jailbreak"
        r = _parse_qwen_output(raw)
        assert r["categories"] == ["Jailbreak"]

    def test_handles_none_sentinel(self) -> None:
        # `None` explicitly means no categories — drop it.
        raw = "Safety: Safe\nCategories: None"
        r = _parse_qwen_output(raw)
        assert r["categories"] == []

    def test_returns_none_safety_on_missing_tag(self) -> None:
        r = _parse_qwen_output("model went off-template, no safety line")
        assert r["safety"] is None
        assert r["categories"] == []

    def test_handles_extra_whitespace(self) -> None:
        raw = "Safety:    Unsafe   \n  Categories:   Violent  ,  PII  "
        r = _parse_qwen_output(raw)
        assert r["safety"] == "unsafe"
        assert set(r["categories"]) == {"Violent", "PII"}


class TestPolicyMappings:
    @pytest.mark.parametrize("policy_cat", list(POLICY_TO_QWEN.keys()))
    def test_every_policy_category_is_a_known_key(self, policy_cat: str) -> None:
        # The dispatcher's CRITERION_TEXT is the source of truth for
        # what categories the dashboard exposes — POLICY_TO_QWEN must
        # cover all of them (even if some map to the empty set).
        from retroguard_classifier.server import CRITERION_TEXT
        assert policy_cat in CRITERION_TEXT

    def test_every_qwen_target_is_in_qwen_taxonomy(self) -> None:
        all_qwen_targets: set[str] = set()
        for s in POLICY_TO_QWEN.values():
            all_qwen_targets.update(s)
        known = set(QWEN_CATEGORIES_INPUT) | set(QWEN_CATEGORIES_OUTPUT)
        assert all_qwen_targets <= known, (
            f"unrecognized Qwen targets: {all_qwen_targets - known}"
        )

    def test_jailbreaking_routes_only_to_jailbreak(self) -> None:
        assert POLICY_TO_QWEN["jailbreaking"] == frozenset({"Jailbreak"})

    def test_violence_routes_to_violent(self) -> None:
        assert "Violent" in POLICY_TO_QWEN["violence"]

    def test_profanity_has_empty_qwen_mapping(self) -> None:
        # Documented limitation: Qwen3Guard isn't natively trained on
        # profanity. Customers needing it must use heavy + custom criteria.
        assert POLICY_TO_QWEN["profanity"] == frozenset()

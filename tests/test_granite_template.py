"""Unit tests for the Granite Guardian prompt template + score parser.

Pin the IBM-spec'd shapes so we don't drift from the model card:
  - Guardian block has the right `<guardian>...<no-think>...</no-think>` envelope
  - `### Criteria:` and `### Scoring Schema:` sections present
  - Chat-template wrap with role markers in the right order
  - Score regex correctly extracts verdicts from `<think>...</think><score>yes|no</score>`
  - Whitespace + casing + missing-tag failure modes handled

These tests don't load the GGUF — pure string + regex logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `retroguard_classifier.server` imports `retroguard_classifier.nsm` which
# tries to ctypes-load `/dev/nsm` at import time on a real enclave. That
# module-level binding is fine (no ioctl until called); but we still guard
# the import so test collection stays clean.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from retroguard_classifier.server import (  # noqa: E402
    CRITERION_TEXT,
    _build_granite_judge_prompt,
    _build_guardian_block,
    _parse_score,
)


class TestGuardianBlock:
    def test_no_think_envelope_is_present(self) -> None:
        block = _build_guardian_block("X criterion")
        assert block.startswith("<guardian>")
        assert "<no-think>" in block
        assert "<score></score>" in block

    def test_criteria_section_present(self) -> None:
        block = _build_guardian_block("X criterion")
        assert "### Criteria: X criterion" in block

    def test_scoring_schema_matches_ibm_template_verbatim(self) -> None:
        """IBM's reference build_guardian_block hardcodes "the last
        assistant's text" — same wording for every direction. Granite
        was trained on this exact phrasing; customizing it per
        direction drifts from the training distribution."""
        block = _build_guardian_block("X criterion")
        assert "### Scoring Schema:" in block
        assert "the last assistant's text" in block
        assert "the user's message" not in block
        assert "return 'yes'" in block and "return 'no'" in block


class TestPrompt:
    def test_input_direction_places_user_text_first(self) -> None:
        prompt = _build_granite_judge_prompt(
            text="hello world",
            direction="input",
            criterion_text="Harm: Universally harmful content.",
        )
        assert "<|start_of_role|>user<|end_of_role|>hello world" in prompt
        # Last turn primes the assistant for the verdict.
        assert prompt.rstrip().endswith("<|start_of_role|>assistant<|end_of_role|>")
        # Schema target is hardcoded per IBM's spec, regardless of direction.
        assert "the last assistant's text" in prompt

    def test_output_direction_places_assistant_text_in_assistant_turn(self) -> None:
        prompt = _build_granite_judge_prompt(
            text="here is the answer",
            direction="output",
            criterion_text="Harm: Universally harmful content.",
        )
        assert "<|start_of_role|>assistant<|end_of_role|>here is the answer" in prompt
        assert "the last assistant's text" in prompt

    def test_prompt_includes_official_instruction(self) -> None:
        prompt = _build_granite_judge_prompt(
            text="x", direction="input", criterion_text="Harm: ..."
        )
        # Sample a stable phrase from IBM's no-think instruction.
        assert "judge agent" in prompt
        assert "without explanation" in prompt
        # And the `<score></score>` self-anchored output marker.
        assert "<score></score>" in prompt


class TestCriterionTaxonomy:
    @pytest.mark.parametrize(
        "category",
        ["harm", "social_bias", "jailbreaking", "violence", "profanity", "unethical_behavior"],
    )
    def test_all_six_ibm_categories_present(self, category: str) -> None:
        assert category in CRITERION_TEXT
        text = CRITERION_TEXT[category]
        # Each entry leads with the human-readable name + colon.
        assert text.split(":", 1)[0].strip().lower().replace(" ", "_") == category


class TestScoreParser:
    def test_parses_yes(self) -> None:
        raw = "<think>\n</think>\n<score>yes</score>"
        assert _parse_score(raw) == "yes"

    def test_parses_no(self) -> None:
        raw = "<think>\n</think>\n<score>no</score>"
        assert _parse_score(raw) == "no"

    def test_parses_with_think_content(self) -> None:
        raw = (
            "<think>\nThe user's text describes building a bomb. This is harmful.\n</think>\n"
            "<score>yes</score>"
        )
        assert _parse_score(raw) == "yes"

    def test_parses_with_whitespace_and_casing(self) -> None:
        raw = "<think></think><score>  YES  </score>"
        assert _parse_score(raw) == "yes"

    def test_returns_none_on_missing_score_tag(self) -> None:
        raw = "<think>some reasoning but no score block</think>"
        assert _parse_score(raw) is None

    def test_returns_none_on_unrecognized_score_value(self) -> None:
        # Granite is constrained to yes/no; anything else means the model
        # drifted and the verdict can't be trusted.
        raw = "<think></think><score>maybe</score>"
        assert _parse_score(raw) is None

    def test_returns_none_on_empty_input(self) -> None:
        assert _parse_score("") is None
        assert _parse_score("   ") is None

    def test_handles_multiline_think_block(self) -> None:
        raw = (
            "<think>\nLine 1\nLine 2\n  with indentation\n</think>\n"
            "<score>no</score>"
        )
        assert _parse_score(raw) == "no"

    def test_parses_when_close_tag_consumed_by_stop_token(self) -> None:
        """llama-cpp's stop-token match eats `</score>` from the output —
        Granite's actual on-wire response looks like `<think>\\n</think>\\n<score> yes`
        with no closing tag. Parser must cope.
        """
        for raw, expected in (
            ("<think>\n</think>\n<score> yes", "yes"),
            ("<think>\n</think>\n<score> no", "no"),
            ("<think>\n</think>\n<score>YES", "yes"),
        ):
            assert _parse_score(raw) == expected, raw

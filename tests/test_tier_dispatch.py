"""Tier dispatch tests for TieredClassifier.

Exercise the dispatch matrix without loading any GGUFs:
  - protection_effort=fast   → qwen_06b
  - protection_effort=expert → qwen_4b
  - protection_effort=heavy  → qwen_8b
  - any tier + custom_criteria → granite (regardless of tier)
  - empty categories AND no custom_criteria → noop fast-path

Mocks substitute fake `classify_one` / `classify_native` methods so
we can inject deterministic verdicts and inspect which engine the
dispatcher chose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from retroguard_classifier.server import TieredClassifier  # noqa: E402


def _qwen_mock(name: str, *, safety: str = "safe", categories: list[str] | None = None) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.classify_native = MagicMock(
        return_value={"safety": safety, "categories": categories or [], "raw": ""}
    )
    return m


def _granite_mock(*, score: str = "no") -> MagicMock:
    m = MagicMock()
    m.name = "granite"
    m.classify_one = MagicMock(return_value={"score": score, "raw": ""})
    return m


def _classifier(
    *,
    granite_score: str = "no",
    qwen_06b_safety: str = "safe",
    qwen_4b_safety: str = "safe",
    qwen_8b_safety: str = "safe",
    qwen_categories: list[str] | None = None,
) -> tuple[TieredClassifier, dict[str, MagicMock]]:
    granite = _granite_mock(score=granite_score)
    q06 = _qwen_mock("qwen_06b", safety=qwen_06b_safety, categories=qwen_categories)
    q4 = _qwen_mock("qwen_4b", safety=qwen_4b_safety, categories=qwen_categories)
    q8 = _qwen_mock("qwen_8b", safety=qwen_8b_safety, categories=qwen_categories)
    c = TieredClassifier(granite=granite, qwen_06b=q06, qwen_4b=q4, qwen_8b=q8)
    return c, {"granite": granite, "qwen_06b": q06, "qwen_4b": q4, "qwen_8b": q8}


class TestTierRouting:
    def test_fast_routes_to_qwen_06b(self) -> None:
        c, mocks = _classifier()
        c.classify(text="x", direction="input", categories=["harm"], protection_effort="fast")
        mocks["qwen_06b"].classify_native.assert_called_once()
        mocks["qwen_4b"].classify_native.assert_not_called()
        mocks["qwen_8b"].classify_native.assert_not_called()
        mocks["granite"].classify_one.assert_not_called()

    def test_expert_routes_to_qwen_4b(self) -> None:
        c, mocks = _classifier()
        c.classify(text="x", direction="input", categories=["harm"], protection_effort="expert")
        mocks["qwen_4b"].classify_native.assert_called_once()
        mocks["qwen_06b"].classify_native.assert_not_called()
        mocks["qwen_8b"].classify_native.assert_not_called()
        mocks["granite"].classify_one.assert_not_called()

    def test_heavy_routes_to_qwen_8b(self) -> None:
        c, mocks = _classifier()
        c.classify(text="x", direction="input", categories=["harm"], protection_effort="heavy")
        mocks["qwen_8b"].classify_native.assert_called_once()
        mocks["qwen_06b"].classify_native.assert_not_called()
        mocks["qwen_4b"].classify_native.assert_not_called()
        mocks["granite"].classify_one.assert_not_called()

    def test_unknown_tier_falls_back_to_expert(self) -> None:
        c, mocks = _classifier()
        c.classify(text="x", direction="input", categories=["harm"], protection_effort="ultra")
        # `expert` is the documented safe fallback.
        mocks["qwen_4b"].classify_native.assert_called_once()


class TestCustomCriteriaRouting:
    @pytest.mark.parametrize("tier", ["fast", "expert", "heavy"])
    def test_custom_criteria_force_granite_regardless_of_tier(self, tier: str) -> None:
        c, mocks = _classifier()
        c.classify(
            text="x",
            direction="input",
            categories=["harm"],
            protection_effort=tier,
            custom_criteria=[{"id": "c1", "text": "no internal codenames"}],
        )
        # Per-criterion eval: one Granite call per criterion (1 builtin + 1
        # custom = 2 here, since none returned 'yes' to short-circuit).
        assert mocks["granite"].classify_one.call_count >= 1
        # And no Qwen engine got the request.
        for label in ("qwen_06b", "qwen_4b", "qwen_8b"):
            mocks[label].classify_native.assert_not_called()

    def test_empty_custom_criteria_text_does_not_pin_to_granite(self) -> None:
        c, mocks = _classifier()
        c.classify(
            text="x",
            direction="input",
            categories=["harm"],
            protection_effort="fast",
            # Whitespace-only text should be treated as no custom criteria.
            custom_criteria=[{"id": "c1", "text": "   "}],
        )
        mocks["qwen_06b"].classify_native.assert_called_once()
        mocks["granite"].classify_one.assert_not_called()


class TestNoopFastPath:
    def test_no_categories_and_no_custom_criteria_returns_noop(self) -> None:
        c, mocks = _classifier()
        r = c.classify(text="x", direction="input", categories=[], protection_effort="expert")
        assert r["verdict"] == "safe"
        assert r["engine"] == "noop"
        # No engine should have been touched.
        for m in mocks.values():
            assert (
                not m.classify_one.called if hasattr(m, "classify_one") else True
            )
            assert (
                not m.classify_native.called if hasattr(m, "classify_native") else True
            )


class TestQwenDispatchSemantics:
    def test_qwen_safe_returns_safe(self) -> None:
        c, _ = _classifier(qwen_4b_safety="safe")
        r = c.classify(text="x", direction="input", categories=["harm"])
        assert r["verdict"] == "safe"
        assert r["label"] is None
        assert r["engine"] == "qwen_4b"

    def test_qwen_unsafe_with_matching_category_blocks(self) -> None:
        c, _ = _classifier(qwen_4b_safety="unsafe", qwen_categories=["Violent"])
        r = c.classify(text="x", direction="input", categories=["violence"])
        assert r["verdict"] == "unsafe"
        assert r["label"] == "violence"
        assert r["per_category"]["violence"] == "yes"

    def test_qwen_unsafe_without_matching_policy_category_does_not_block(self) -> None:
        # Qwen flagged Sexual Content but the policy only enabled `harm` /
        # `violence`. Per the explicit policy, this should NOT block.
        c, _ = _classifier(
            qwen_4b_safety="unsafe", qwen_categories=["Sexual Content or Sexual Acts"]
        )
        r = c.classify(text="x", direction="input", categories=["harm", "violence"])
        assert r["verdict"] == "safe"
        assert r["label"] is None

    def test_qwen_controversial_treated_as_unsafe(self) -> None:
        c, _ = _classifier(
            qwen_4b_safety="controversial", qwen_categories=["Politically Sensitive Topics"]
        )
        # Customer enabled `social_bias` which doesn't map to political —
        # so the controversial verdict has no matching enabled category.
        r = c.classify(text="x", direction="input", categories=["social_bias"])
        assert r["verdict"] == "safe"

    def test_jailbreak_fires_correctly(self) -> None:
        c, _ = _classifier(qwen_4b_safety="unsafe", qwen_categories=["Jailbreak"])
        r = c.classify(text="x", direction="input", categories=["jailbreaking"])
        assert r["verdict"] == "unsafe"
        assert r["label"] == "jailbreaking"


class TestGraniteCustomDispatch:
    """Per-criterion BYOC eval: built-in categories are evaluated FIRST,
    custom rules NEXT, both in declared order. Short-circuits on the
    first 'yes'. Matches IBM's spec — one criterion per
    apply_chat_template call."""

    def test_granite_safe_returns_safe(self) -> None:
        c, _ = _classifier(granite_score="no")
        r = c.classify(
            text="x",
            direction="input",
            categories=["harm"],
            custom_criteria=[{"id": "c1", "text": "no codenames"}],
        )
        assert r["verdict"] == "safe"
        assert r["engine"] == "granite"
        assert r["per_category"]["harm"] == "no"
        assert r["per_category"]["c1"] == "no"

    def test_granite_unsafe_short_circuits_on_first_builtin(self) -> None:
        """All Granite calls return 'yes' — the first criterion (a
        built-in: 'harm') wins, custom rules are marked 'skipped'."""
        c, mocks = _classifier(granite_score="yes")
        r = c.classify(
            text="x",
            direction="input",
            categories=["harm"],
            custom_criteria=[
                {"id": "c-no-codenames", "text": "no codenames"},
                {"id": "c-no-pii", "text": "no PII"},
            ],
        )
        assert r["verdict"] == "unsafe"
        # Builtins evaluated first → 'harm' fires before custom criteria
        # are even visited.
        assert r["label"] == "harm"
        assert r["per_category"]["harm"] == "yes"
        assert r["per_category"]["c-no-codenames"] == "skipped"
        assert r["per_category"]["c-no-pii"] == "skipped"
        # And only ONE Granite call should have happened (the harm one).
        assert mocks["granite"].classify_one.call_count == 1

    def test_granite_unsafe_attributes_to_specific_custom_when_only_custom(self) -> None:
        """No built-ins, multiple custom — first custom criterion wins."""
        c, mocks = _classifier(granite_score="yes")
        r = c.classify(
            text="x",
            direction="input",
            categories=[],
            custom_criteria=[
                {"id": "c-no-codenames", "text": "no codenames"},
                {"id": "c-no-pii", "text": "no PII"},
            ],
        )
        assert r["verdict"] == "unsafe"
        assert r["label"] == "c-no-codenames"
        assert r["per_category"]["c-no-codenames"] == "yes"
        assert r["per_category"]["c-no-pii"] == "skipped"

    def test_granite_unsafe_with_first_builtin_in_list(self) -> None:
        c, _ = _classifier(granite_score="yes")
        r = c.classify(
            text="x",
            direction="input",
            categories=["jailbreaking", "violence"],
            custom_criteria=[{"id": "c1", "text": "rule"}],
        )
        assert r["verdict"] == "unsafe"
        # 'jailbreaking' is first in the list → it wins.
        assert r["label"] == "jailbreaking"

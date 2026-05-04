"""Pinned tests for the v2 default policy category set.

For every category the dashboard exposes — and every back-compat alias —
this suite verifies:

  1. Granite has a CRITERION_TEXT entry (per-criterion eval can run)
  2. Qwen has a POLICY_TO_QWEN entry (Qwen-tier classify can route it)
  3. The Qwen mapping references only valid Qwen native categories
     (so the dispatcher's set-intersection actually fires)
  4. Dispatcher routes the category correctly under each path:
       - Qwen tier, Qwen flagged the right native    → fires
       - Granite-only path (custom criteria present) → fires
       - Hybrid path for the profanity gap           → Granite is called
  5. Aliases produce equivalent behaviour to their canonical names

Each test is small and isolated so a regression in any single
category surfaces a clear failure name (e.g.
`test_pii_qwen_path_fires_on_native_pii`) instead of a vague suite
breakage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from retroguard_classifier.server import (  # noqa: E402
    CRITERION_TEXT,
    POLICY_TO_QWEN,
    QWEN_CATEGORIES_INPUT,
    QWEN_CATEGORIES_OUTPUT,
    TieredClassifier,
)


# ────────────────────────────────────────────────────────────────────
# Constants the dashboard / spec must agree with the dispatcher on.
# ────────────────────────────────────────────────────────────────────
DEFAULT_CATEGORIES: list[str] = [
    "violence",
    "self_harm",
    "sexual_content",
    "pii",
    "hate_speech",
    "profanity",
    "illegal_acts",
    "prompt_injection",
    "copyright_violation",
    "misinformation",
]

LEGACY_ALIASES: list[str] = [
    "harm",
    "social_bias",
    "unethical_behavior",
    "jailbreaking",
]

ALL_KNOWN_CATEGORIES: list[str] = DEFAULT_CATEGORIES + LEGACY_ALIASES

# Every alias resolves to the same Qwen native set as its canonical
# new-name twin. If we ever drift these, customers' verdicts shift
# silently across the rename — block that at test time.
ALIAS_PAIRS: list[tuple[str, str]] = [
    ("jailbreaking", "prompt_injection"),
    ("social_bias", "hate_speech"),
]


# ────────────────────────────────────────────────────────────────────
# Mock fixtures (mirror test_tier_dispatch.py — same shape so the two
# suites stay aligned).
# ────────────────────────────────────────────────────────────────────
def _qwen_mock(
    name: str,
    *,
    safety: str = "safe",
    categories: list[str] | None = None,
) -> MagicMock:
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
    qwen_safety: str = "safe",
    qwen_categories: list[str] | None = None,
) -> tuple[TieredClassifier, dict[str, MagicMock]]:
    granite = _granite_mock(score=granite_score)
    q06 = _qwen_mock("qwen_06b", safety=qwen_safety, categories=qwen_categories)
    q4 = _qwen_mock("qwen_4b", safety=qwen_safety, categories=qwen_categories)
    q8 = _qwen_mock("qwen_8b", safety=qwen_safety, categories=qwen_categories)
    c = TieredClassifier(granite=granite, qwen_06b=q06, qwen_4b=q4, qwen_8b=q8)
    return c, {"granite": granite, "qwen_06b": q06, "qwen_4b": q4, "qwen_8b": q8}


# ────────────────────────────────────────────────────────────────────
# 1. Static guarantees — every category has a path through both
#    classifiers. These run for every category in one go via
#    parametrize so a missing entry is named explicitly in the
#    failure output.
# ────────────────────────────────────────────────────────────────────
class TestStaticGuarantees:
    @pytest.mark.parametrize("category", ALL_KNOWN_CATEGORIES)
    def test_category_has_granite_criterion_text(self, category: str) -> None:
        assert category in CRITERION_TEXT, (
            f"category {category!r} has no CRITERION_TEXT entry — "
            "Granite per-criterion eval would skip it"
        )
        text = CRITERION_TEXT[category]
        assert ":" in text, f"{category!r} criterion text missing 'Title:' prefix"
        assert len(text) >= 30, f"{category!r} criterion text suspiciously short"

    @pytest.mark.parametrize("category", ALL_KNOWN_CATEGORIES)
    def test_category_has_qwen_mapping(self, category: str) -> None:
        assert category in POLICY_TO_QWEN, (
            f"category {category!r} has no POLICY_TO_QWEN entry — "
            "Qwen tier would silently drop verdicts for it"
        )

    @pytest.mark.parametrize("category", ALL_KNOWN_CATEGORIES)
    def test_qwen_mapping_uses_valid_native_categories(self, category: str) -> None:
        valid = set(QWEN_CATEGORIES_INPUT) | set(QWEN_CATEGORIES_OUTPUT)
        for native in POLICY_TO_QWEN[category]:
            assert native in valid, (
                f"category {category!r} maps to {native!r}, which is not "
                "in Qwen's published taxonomy — Qwen never emits this "
                "label so the category would never fire"
            )

    def test_profanity_is_the_only_qwen_gap(self) -> None:
        """Profanity has no Qwen native equivalent — empty mapping is
        intentional and triggers the hybrid Granite call. Any other
        empty mapping is almost certainly a bug."""
        gaps = [c for c, m in POLICY_TO_QWEN.items() if not m]
        assert gaps == ["profanity"], (
            f"unexpected Qwen-gap categories: {gaps}. Either map them "
            "to a Qwen native, or accept that they only fire on the "
            "Granite (heavy + custom) path."
        )


# ────────────────────────────────────────────────────────────────────
# 2. Per-category dispatcher tests — Qwen path. Each one mocks Qwen
#    flagging the category's native, then asserts the policy
#    category fires with the right label.
# ────────────────────────────────────────────────────────────────────
QWEN_NATIVE_FOR_TEST: dict[str, str] = {
    "violence": "Violent",
    "self_harm": "Suicide & Self-Harm",
    "sexual_content": "Sexual Content or Sexual Acts",
    "pii": "PII",
    "hate_speech": "Unethical Acts",
    "illegal_acts": "Non-violent Illegal Acts",
    "prompt_injection": "Jailbreak",
    "copyright_violation": "Copyright Violation",
    "misinformation": "Politically Sensitive Topics",
    # Aliases that have a single direct Qwen native.
    "social_bias": "Unethical Acts",
    "jailbreaking": "Jailbreak",
}


class TestQwenPath:
    @pytest.mark.parametrize(
        "category,native",
        list(QWEN_NATIVE_FOR_TEST.items()),
    )
    def test_qwen_native_hit_fires_policy_category(
        self, category: str, native: str
    ) -> None:
        c, mocks = _classifier(qwen_safety="unsafe", qwen_categories=[native])
        r = c.classify(text="x", direction="input", categories=[category])
        assert r["verdict"] == "unsafe", (
            f"{category!r}: Qwen flagged {native!r} but verdict was safe — "
            "POLICY_TO_QWEN mapping likely missing or wrong"
        )
        assert r["label"] == category
        assert r["per_category"][category] == "yes"
        # Engine should be the Qwen tier we picked (default expert → 4B).
        assert mocks["qwen_4b"].classify_native.called
        assert not mocks["granite"].classify_one.called

    def test_qwen_safe_returns_safe_for_each_category(self) -> None:
        # One Qwen call should produce a clean 'safe' verdict regardless
        # of how many categories the customer enabled.
        c, _ = _classifier(qwen_safety="safe")
        r = c.classify(
            text="x", direction="input", categories=DEFAULT_CATEGORIES
        )
        assert r["verdict"] == "safe"
        assert r["label"] is None


# ────────────────────────────────────────────────────────────────────
# 3. Granite path — custom-criteria case. Granite per-criterion eval
#    is exercised when the policy carries any custom rule. Verifies
#    each built-in category gets evaluated against its CRITERION_TEXT
#    string when present alongside customs.
# ────────────────────────────────────────────────────────────────────
class TestGranitePath:
    @pytest.mark.parametrize("category", DEFAULT_CATEGORIES + LEGACY_ALIASES)
    def test_granite_per_criterion_fires_on_yes(self, category: str) -> None:
        c, mocks = _classifier(granite_score="yes")
        r = c.classify(
            text="x",
            direction="input",
            categories=[category],
            custom_criteria=[{"id": "noise", "text": "no codenames"}],
        )
        assert r["verdict"] == "unsafe"
        assert r["label"] == category, (
            f"{category!r}: Granite returned 'yes' but per-criterion "
            "eval failed to attribute the block to the built-in category"
        )
        assert r["engine"] == "granite"
        # First Granite call should be for the built-in (built-ins
        # evaluated before custom rules per IBM's spec).
        assert mocks["granite"].classify_one.call_count >= 1

    @pytest.mark.parametrize("category", DEFAULT_CATEGORIES + LEGACY_ALIASES)
    def test_granite_uses_canonical_criterion_text(self, category: str) -> None:
        """The criterion_text Granite sees must come from CRITERION_TEXT
        verbatim — silent rewrites here would drift Granite's verdict
        accuracy from the model card baseline."""
        c, mocks = _classifier(granite_score="no")
        c.classify(
            text="x",
            direction="input",
            categories=[category],
            custom_criteria=[{"id": "c1", "text": "rule"}],
        )
        # First built-in call has criterion_text = CRITERION_TEXT[cat]
        first_call = mocks["granite"].classify_one.call_args_list[0]
        assert first_call.kwargs["criterion_text"] == CRITERION_TEXT[category]


# ────────────────────────────────────────────────────────────────────
# 4. Hybrid dispatch — profanity is the only gap; verify the
#    dispatcher actually issues a Granite call for it even when no
#    custom criteria are attached and the policy enables only
#    Qwen-native categories besides profanity.
# ────────────────────────────────────────────────────────────────────
class TestProfanityHybrid:
    def test_profanity_alone_routes_to_granite_on_qwen_tier(self) -> None:
        c, mocks = _classifier(granite_score="yes", qwen_safety="safe")
        r = c.classify(
            text="x", direction="input", categories=["profanity"]
        )
        # Granite must have been called for profanity (the gap).
        assert mocks["granite"].classify_one.called, (
            "profanity should hybrid-dispatch to Granite when on a "
            "Qwen tier — the empty POLICY_TO_QWEN mapping is the trigger"
        )
        # And the Granite forward pass should use profanity's
        # CRITERION_TEXT, not some other criterion.
        first = mocks["granite"].classify_one.call_args_list[0]
        assert first.kwargs["criterion_text"] == CRITERION_TEXT["profanity"]
        assert r["verdict"] == "unsafe"
        assert r["label"] == "profanity"

    def test_mixed_categories_use_both_engines(self) -> None:
        # Customer enables violence (Qwen-native) + profanity (gap).
        # Both engines should run; verdict aggregates.
        c, mocks = _classifier(
            granite_score="yes",     # profanity will fire on Granite
            qwen_safety="safe",      # violence won't fire on Qwen
        )
        r = c.classify(
            text="x",
            direction="input",
            categories=["violence", "profanity"],
        )
        assert mocks["qwen_4b"].classify_native.called
        assert mocks["granite"].classify_one.called
        assert r["verdict"] == "unsafe"
        assert r["label"] == "profanity"
        assert r["engine"].startswith("hybrid:")

    def test_profanity_safe_when_neither_engine_fires(self) -> None:
        c, mocks = _classifier(granite_score="no", qwen_safety="safe")
        r = c.classify(
            text="x",
            direction="input",
            categories=["violence", "profanity"],
        )
        assert r["verdict"] == "safe"
        assert r["label"] is None
        # Both engines still queried because the dispatch split is
        # static (depends on POLICY_TO_QWEN, not on results).
        assert mocks["qwen_4b"].classify_native.called
        assert mocks["granite"].classify_one.called


# ────────────────────────────────────────────────────────────────────
# 5. Alias parity — legacy IDs route identically to their modern
#    twins. Drift here would silently change verdicts for customers
#    who've kept the old ID in their policies.
# ────────────────────────────────────────────────────────────────────
class TestAliasParity:
    @pytest.mark.parametrize("alias,canonical", ALIAS_PAIRS)
    def test_aliased_qwen_mapping_matches_canonical(
        self, alias: str, canonical: str
    ) -> None:
        assert POLICY_TO_QWEN[alias] == POLICY_TO_QWEN[canonical], (
            f"alias {alias!r} maps to {POLICY_TO_QWEN[alias]} but its "
            f"canonical twin {canonical!r} maps to {POLICY_TO_QWEN[canonical]}. "
            "These must stay in sync — aliases exist precisely so the "
            "verdict shape doesn't shift across the rename."
        )

    @pytest.mark.parametrize("alias,canonical", ALIAS_PAIRS)
    def test_aliased_dispatcher_verdict_matches_canonical(
        self, alias: str, canonical: str
    ) -> None:
        # Same Qwen flag → same verdict whether customer enabled the
        # alias or the canonical.
        native_for_canonical = QWEN_NATIVE_FOR_TEST[canonical]

        c1, _ = _classifier(qwen_safety="unsafe", qwen_categories=[native_for_canonical])
        r_alias = c1.classify(text="x", direction="input", categories=[alias])

        c2, _ = _classifier(qwen_safety="unsafe", qwen_categories=[native_for_canonical])
        r_canon = c2.classify(text="x", direction="input", categories=[canonical])

        assert r_alias["verdict"] == r_canon["verdict"]
        # The label intentionally fires under whichever ID the customer
        # enabled — preserves attribution semantics across the rename.
        assert r_alias["label"] == alias
        assert r_canon["label"] == canonical

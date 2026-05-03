"""In-enclave classifier server.

Runs inside the Nitro Enclave. Hosts FOUR GGUFs that the parent
streams over vsock at boot:

    granite      — granite-guardian-4.1-8b   (i1-Q4_K_S, ~4.7 GB)
                   Used ONLY when the request carries custom NL
                   criteria — Granite was trained on a custom-criteria
                   schema that other guard models can't replicate.
    qwen_06b     — Qwen3Guard-Gen-0.6B       (Q4_K_M,    ~0.4 GB)  → tier `fast`
    qwen_4b      — Qwen3Guard-Gen-4B         (Q4_K_M,    ~2.5 GB)  → tier `expert`
    qwen_8b      — Qwen3Guard-Gen-8B         (Q4_K_M,    ~5.1 GB)  → tier `heavy`

Each blob's SHA-256 is baked into the EIF (and therefore PCR0).
The parent CAN read the bytes (all four checkpoints are public),
but cannot substitute a different file: the enclave refuses any
blob whose digest doesn't match the baked constant.

Wire protocol:
  * Port 5006 (load): one connection. Header JSON declares which
    `models` are coming (in order); for each, an [8-byte BE length]
    [bytes] frame follows. KMS mode adds an attestation handshake
    before the framed blobs (see `_negotiate_kms_data_key`).
  * Port 5005 (classify): newline-delimited JSON.
        Request:  {"op":"classify","request_id":"...","text":"...",
                   "direction":"input|output","categories":["harm",...],
                   "protection_effort":"fast|expert|heavy",
                   "custom_criteria":[{"id":"...","text":"..."}]}
        Response: {"request_id":"...","verdict":"safe|unsafe","label":"<cat>"|null,
                   "per_category":{"<cat>":"yes|no",...},"engine":"granite|qwen_xx",
                   "latency_ms":...}
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retroguard_classifier import nsm
from retroguard_classifier.upstream import (
    DEFAULT_UPSTREAM_PORTS,
    HttpReader,
    HttpError,
    open_tls_over_vsock,
    send_request,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enclave")

LOAD_PORT = int(os.environ.get("RG_LOAD_PORT", "5006"))
CLASSIFY_PORT = int(os.environ.get("RG_VSOCK_PORT", "5005"))
MODEL_DIR = Path(os.environ.get("RG_MODEL_DIR", "/tmp/models"))
MODEL_CTX = int(os.environ.get("RG_MODEL_CTX", "4096"))
MODEL_THREADS = int(os.environ.get("RG_MODEL_THREADS", "32"))


# --------------------------------------------------------------------------
# Model registry — each tier maps to a labelled GGUF on disk + a baked
# SHA-256 (env-injected at EIF build time, therefore measured by PCR0).
# `protection_effort` request field selects qwen_06b / qwen_4b / qwen_8b;
# any request with non-empty `custom_criteria` is forced onto granite
# regardless of the requested tier.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    label: str
    filename: str
    sha256: str  # lowercase hex; "" disables the model (load will refuse)


def _spec(label: str, filename: str, env_var: str) -> ModelSpec:
    return ModelSpec(
        label=label,
        filename=filename,
        sha256=os.environ.get(env_var, "").strip().lower(),
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "granite":  _spec("granite",  "granite-guardian.gguf", "RG_GRANITE_SHA256"),
    "qwen_06b": _spec("qwen_06b", "qwen3guard-gen-0.6b.gguf", "RG_QWEN_06B_SHA256"),
    "qwen_4b":  _spec("qwen_4b",  "qwen3guard-gen-4b.gguf",  "RG_QWEN_4B_SHA256"),
    "qwen_8b":  _spec("qwen_8b",  "qwen3guard-gen-8b.gguf",  "RG_QWEN_8B_SHA256"),
}

# Backwards-compat: kept so existing callers / tests can still reference the
# Granite path directly. `MODEL_SPECS["granite"]` is the source of truth.
GRANITE_PATH = MODEL_DIR / MODEL_SPECS["granite"].filename


def _model_path(label: str) -> Path:
    return MODEL_DIR / MODEL_SPECS[label].filename


# --------------------------------------------------------------------------
# Granite Guardian 4.1 prompt template (per IBM model card, no-think mode).
# Used only when the request carries custom NL criteria — Granite was
# trained on the `<guardian>... ### Criteria: ... ### Scoring Schema: ...`
# schema and gives the strongest signal on free-form rules.
# --------------------------------------------------------------------------

GUARDIAN_INSTRUCTION_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)

# Pre-baked criterion text — verbatim phrasings that map onto IBM's
# six trained safety categories.
CRITERION_TEXT: dict[str, str] = {
    "harm": (
        "Harm: Universally harmful content across all contexts."
    ),
    "social_bias": (
        "Social Bias: Systemic prejudice against groups; stereotypes and "
        "discriminatory attitudes."
    ),
    "jailbreaking": (
        "Jailbreaking: Deliberate evasion of an AI system's built-in safeguards "
        "or ethical boundaries; manipulation into providing restricted content."
    ),
    "violence": (
        "Violence: Physical harm depictions, assault, self-harm, threats, "
        "or intimidation."
    ),
    "profanity": (
        "Profanity: Offensive or socially unacceptable language; curse words, "
        "swear words (excludes slurs)."
    ),
    "unethical_behavior": (
        "Unethical Behavior: Fraud, theft, abuse of power; moral or "
        "professional violations (excludes physical violence)."
    ),
}


def _build_guardian_block(criterion_text: str) -> str:
    """Assemble the IBM-spec'd <guardian> message body.

    Scoring schema text is hardcoded to 'the last assistant's text' to
    match IBM's reference build_guardian_block from the model card —
    that's the exact phrasing Granite was trained on, and it's used
    even for input-direction (USER message) checks. We had earlier
    customized the target per direction; that drifted from training and
    measurably hurt accuracy on BYOC criteria.
    """
    return (
        f"<guardian>{GUARDIAN_INSTRUCTION_NOTHINK}\n\n"
        f"### Criteria: {criterion_text}\n\n"
        "### Scoring Schema: If the last assistant's text meets the criteria, "
        "return 'yes'; otherwise, return 'no'."
    )


def _build_granite_judge_prompt(text: str, direction: str, criterion_text: str) -> str:
    """Build the official Granite chat-template-formatted prompt.

    Mirrors IBM's reference `apply_chat_template(..., add_generation_prompt=True)`
    output for both input + output direction — the difference is only
    where in the conversation the text-being-judged sits. The guardian
    block is always the LAST user message regardless of direction.
    """
    guardian = _build_guardian_block(criterion_text)
    if direction == "input":
        return (
            f"<|start_of_role|>user<|end_of_role|>{text}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{guardian}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )
    return (
        "<|start_of_role|>user<|end_of_role|>(prior user turn)<|end_of_text|>\n"
        f"<|start_of_role|>assistant<|end_of_role|>{text}<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{guardian}<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )


_SCORE_RE = re.compile(r"<score>\s*(yes|no)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_score(raw: str) -> str | None:
    """Pull the verdict out of `<think>...</think><score>yes|no(</score>)?`."""
    cleaned = _THINK_RE.sub("", raw or "", count=1).strip()
    m = _SCORE_RE.search(cleaned)
    if not m:
        return None
    return m.group(1).lower()


# --------------------------------------------------------------------------
# Qwen3Guard-Gen prompt template + parser.
#
# These templates are pasted (with empty <think>) FROM the model's
# tokenizer_config.json `chat_template` so the on-wire prompt matches
# what the model was fine-tuned on. Anything else and the model emits
# free-form text — the parser sees no `Safety:` line and the
# dispatcher treats it as safe (false negative across the board).
#
# Output (model emits this verbatim after the assistant prime):
#     Safety: Safe|Unsafe|Controversial
#     Categories: <comma-separated list>|None
#     [Refusal: Yes|No]                 # response moderation only
# --------------------------------------------------------------------------

# Qwen3Guard taxonomy (from the model card). Input adds Jailbreak.
QWEN_CATEGORIES_OUTPUT = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
)
QWEN_CATEGORIES_INPUT = QWEN_CATEGORIES_OUTPUT + ("Jailbreak",)

# Map from policy categories (the customer-facing taxonomy that the
# dashboard exposes) to the Qwen taxonomy. A policy category fires
# iff Qwen's emitted Categories list intersects this set.
#
# `profanity` has no native Qwen3Guard category — customers needing
# strict profanity moderation should use heavy + custom criteria
# (Granite path). Surfaced at policy-edit time, not at runtime.
POLICY_TO_QWEN: dict[str, frozenset[str]] = {
    "harm": frozenset({
        "Violent", "Suicide & Self-Harm", "Non-violent Illegal Acts",
    }),
    "violence": frozenset({"Violent"}),
    "jailbreaking": frozenset({"Jailbreak"}),
    "social_bias": frozenset({"Unethical Acts"}),
    "profanity": frozenset(),
    "unethical_behavior": frozenset({
        "Unethical Acts", "Non-violent Illegal Acts",
    }),
}


_QWEN_INPUT_PREAMBLE = (
    "<|im_start|>user\n# Task:\n"
    "1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.\n"
    "2) Identify the category of unsafe content found.\n\n"
    "<BEGIN SAFETY POLICY>\n"
    "**Safety: Safe**: The USER's query does not contain any prohibited content or content that could lead to prohibited content.\n"
    "**Safety: Unsafe**: The USER's query contains prohibited content.\n"
    "**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.\n"
    "<END SAFETY POLICY>\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "Violent.\n"
    "Non-violent Illegal Acts.\n"
    "Sexual Content or Sexual Acts.\n"
    "PII.\n"
    "Suicide & Self-Harm.\n"
    "Unethical Acts.\n"
    "Politically Sensitive Topics.\n"
    "Copyright Violation.\n"
    "Jailbreak.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\n\n"
)
_QWEN_INPUT_TRAILER = (
    "\n\n<END CONVERSATION>\n\n"
    "Provide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:\n"
    " - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n"
    " - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

_QWEN_OUTPUT_PREAMBLE = (
    "<|im_start|>user\n# Task:\n"
    "1) Evaluate whether THE LAST ASSISTANT's response belongs to safe or unsafe content based on our safety policy.\n"
    "2) Identify the category of unsafe content found.\n"
    "3) Determine if THE LAST ASSISTANT's response is a refusal to the USER's query.\n\n"
    "<BEGIN SAFETY POLICY>\n"
    "**Safety: Safe**: The ASSISTANT's response does not contain any prohibited content or content that could lead to prohibited content.\n"
    "**Safety: Unsafe**: The ASSISTANT's response contains prohibited content.\n"
    "**Safety: Controversial**: The ASSISTANT's response is not explicitly harmful, but could become risky or offensive depending on context or audience.\n"
    "<END SAFETY POLICY>\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "Violent.\n"
    "Non-violent Illegal Acts.\n"
    "Sexual Content or Sexual Acts.\n"
    "PII.\n"
    "Suicide & Self-Harm.\n"
    "Unethical Acts.\n"
    "Politically Sensitive Topics.\n"
    "Copyright Violation.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN REFUSAL POLICY>\n"
    "**Refusal: Yes**: The ASSISTANT refused to answer or did not comply with the USER's request.\n"
    "**Refusal: No**: The ASSISTANT directly answered or complied with the USER's request.\n"
    "<END REFUSAL POLICY>\n\n"
    "<BEGIN CONVERSATION>\n\n"
)
_QWEN_OUTPUT_TRAILER = (
    "\n\n<END CONVERSATION>\n\n"
    "Provide your safety assessment for ONLY THE LAST **ASSISTANT's response** in the above conversation:\n"
    " - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n"
    " - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.\n"
    " - The third line must be one of: 'Refusal: Yes', 'Refusal: No'.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _build_qwen_judge_prompt(
    text: str, direction: str, prior_user_text: str = "(prior user turn)"
) -> str:
    """Build the exact prompt Qwen3Guard-Gen was fine-tuned on.

    Pasted from the model's tokenizer chat_template — the safety policy +
    category list MUST be in the prompt or the model emits free-form
    text that the parser silently treats as safe.
    """
    if direction == "input":
        return f"{_QWEN_INPUT_PREAMBLE}USER: {text}{_QWEN_INPUT_TRAILER}"
    return (
        f"{_QWEN_OUTPUT_PREAMBLE}"
        f"USER: {prior_user_text}\n\n"
        f"ASSISTANT: {text}"
        f"{_QWEN_OUTPUT_TRAILER}"
    )


_QWEN_SAFETY_RE = re.compile(
    r"Safety\s*:\s*(Safe|Unsafe|Controversial)\b", re.IGNORECASE
)
_QWEN_CATEGORIES_RE = re.compile(
    r"Categor(?:y|ies)\s*:\s*([^\n\r]+)", re.IGNORECASE
)
_QWEN_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_qwen_output(raw: str) -> dict[str, Any]:
    """Extract {safety, categories[]} from Qwen3Guard-Gen output.

    The prompt pre-fills `<think>\\n\\n</think>` so the model normally
    skips it, but defensively strip any think block we encounter before
    matching `Safety:` / `Categories:`.
    """
    text = (raw or "").strip()
    text = _QWEN_THINK_RE.sub("", text, count=1)
    safety_m = _QWEN_SAFETY_RE.search(text)
    cats_m = _QWEN_CATEGORIES_RE.search(text)
    safety = safety_m.group(1).lower() if safety_m else None
    raw_cats = cats_m.group(1) if cats_m else ""
    raw_cats = raw_cats.strip().strip("[]")
    parts = [c.strip().strip("'\"") for c in raw_cats.split(",")]
    cats = [c for c in parts if c and c.lower() != "none"]
    return {"safety": safety, "categories": cats}


# --------------------------------------------------------------------------
# Engines: each wraps one GGUF loaded via llama-cpp-python.
# --------------------------------------------------------------------------

class _LlamaEngineBase:
    """Loads a GGUF and exposes a single completion call.

    Shared base so Granite + Qwen engines share the loader and don't
    re-implement llama-cpp boilerplate.
    """

    def __init__(self, *, name: str, model_path: Path, max_tokens: int, stop: list[str]) -> None:
        log.info(
            "loading %s from %s (ctx=%d, threads=%d)",
            name, model_path, MODEL_CTX, MODEL_THREADS,
        )
        from llama_cpp import Llama  # type: ignore[import-not-found]

        self.name = name
        self._max_tokens = max_tokens
        self._stop = stop
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=MODEL_CTX,
            n_threads=MODEL_THREADS,
            n_batch=512,
            verbose=False,
        )
        log.info("loaded %s", name)

    def _generate(self, prompt: str) -> str:
        out = self._llm(
            prompt,
            max_tokens=self._max_tokens,
            temperature=0.0,
            stop=self._stop,
        )
        return (out.get("choices", [{}])[0].get("text") or "").strip()


class GraniteGuardEngine(_LlamaEngineBase):
    """Granite Guardian 4.1 — per-criterion yes/no judge.

    `</score>` in the stop set so generation halts as soon as the verdict
    lands (~10 tokens). MAX_TOKENS=16 gives a safety margin without
    paying for runaway generation.
    """

    STOP_TOKENS = ["</score>", "<|end_of_text|>", "</s>"]
    MAX_TOKENS = 16

    def __init__(self, model_path: Path) -> None:
        super().__init__(
            name="granite",
            model_path=model_path,
            max_tokens=self.MAX_TOKENS,
            stop=self.STOP_TOKENS,
        )

    def classify_one(self, *, text: str, direction: str, criterion_text: str) -> dict[str, Any]:
        prompt = _build_granite_judge_prompt(text, direction, criterion_text)
        raw = self._generate(prompt)
        score = _parse_score(raw)
        if score is None:
            log.warning(
                "granite verdict unparseable; treating as 'no' (safe). raw=%r", raw[:200]
            )
        return {"score": score or "no", "raw": raw[:200]}


class QwenGuardEngine(_LlamaEngineBase):
    """Qwen3Guard-Gen — single-pass safety classifier.

    Output ≈ "Safety: Unsafe\\nCategories: Violent, Jailbreak\\nRefusal: No"
    — well under 64 tokens. `<|im_end|>` is the natural stop.
    """

    STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
    # Output is up to three lines: "Safety: Unsafe\nCategories: A, B, C, D\nRefusal: No"
    # ~50 tokens worst-case; 96 leaves margin for long category strings.
    MAX_TOKENS = 96

    def __init__(self, *, name: str, model_path: Path) -> None:
        super().__init__(
            name=name,
            model_path=model_path,
            max_tokens=self.MAX_TOKENS,
            stop=self.STOP_TOKENS,
        )

    def classify_native(self, *, text: str, direction: str) -> dict[str, Any]:
        prompt = _build_qwen_judge_prompt(text, direction)
        raw = self._generate(prompt)
        parsed = _parse_qwen_output(raw)
        return {**parsed, "raw": raw[:300]}


# --------------------------------------------------------------------------
# BYOC dispatch helpers. Per IBM's spec each criterion goes through its
# own `apply_chat_template` call — combining N criteria into one bulleted
# prompt drifts from the training distribution and degrades accuracy.
# We OR per-criterion calls at the application layer instead.
# --------------------------------------------------------------------------

def _named_custom_criteria(
    custom_criteria: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """Strip empties + canonicalize the (id, text) pairs."""
    out: list[tuple[str, str]] = []
    for c in custom_criteria:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        out.append((c.get("id") or "custom", text))
    return out


# --------------------------------------------------------------------------
# TieredClassifier — dispatches by (protection_effort, custom_criteria).
# --------------------------------------------------------------------------

class TieredClassifier:
    """Routes each `op:classify` to the right engine.

    Dispatch matrix:
        custom_criteria non-empty → Granite (regardless of tier; we
                                    auto-promote to heavy at the parent
                                    too — this is the safety net)
        tier=fast,   no custom    → Qwen 0.6B
        tier=expert, no custom    → Qwen 4B
        tier=heavy,  no custom    → Qwen 8B

    Customer-facing categories (`harm`, `violence`, `jailbreaking`, ...)
    are mapped onto each engine's native taxonomy — Granite uses the
    IBM-trained criterion text; Qwen uses POLICY_TO_QWEN.

    Eager-load on construction. Memory budget at c7i.12xlarge / 80 GiB
    enclave is comfortable: ~14 GiB working set for all four engines.
    """

    def __init__(
        self,
        *,
        granite: GraniteGuardEngine,
        qwen_06b: QwenGuardEngine,
        qwen_4b: QwenGuardEngine,
        qwen_8b: QwenGuardEngine,
    ) -> None:
        self.granite = granite
        self.qwen_06b = qwen_06b
        self.qwen_4b = qwen_4b
        self.qwen_8b = qwen_8b
        log.info("tiered classifier ready (4 engines loaded)")

    def _select_qwen(self, tier: str) -> QwenGuardEngine:
        if tier == "fast":
            return self.qwen_06b
        if tier == "heavy":
            return self.qwen_8b
        # `expert` is the default and the safe fallback for unknown tiers.
        return self.qwen_4b

    def classify(
        self,
        *,
        text: str,
        direction: str,
        categories: list[str],
        protection_effort: str = "expert",
        custom_criteria: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        custom = [c for c in (custom_criteria or []) if (c.get("text") or "").strip()]
        valid_builtin = [c for c in categories if c in CRITERION_TEXT]
        unknown = [c for c in categories if c not in CRITERION_TEXT]
        for c in unknown:
            log.warning("unknown criterion category: %r — skipping", c)

        if not valid_builtin and not custom:
            return {
                "verdict": "safe", "label": None, "per_category": {},
                "engine": "noop", "mode": "noop",
            }

        # Custom criteria → always Granite (the only engine trained on
        # arbitrary natural-language rules). Customer's tier choice is
        # respected at the parent's policy gate; here we just route.
        if custom:
            return self._classify_granite_custom(
                text=text,
                direction=direction,
                builtin_categories=valid_builtin,
                custom_criteria=custom,
            )

        return self._classify_qwen(
            engine=self._select_qwen(protection_effort),
            text=text,
            direction=direction,
            categories=valid_builtin,
        )

    # --- Granite path (custom criteria) -----------------------------------

    def _classify_granite_custom(
        self,
        *,
        text: str,
        direction: str,
        builtin_categories: list[str],
        custom_criteria: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Per-criterion Granite evaluation matching IBM's BYOC spec.

        Each criterion (built-in category and customer custom rule) gets
        its own forward pass with its own `apply_chat_template` prompt.
        Short-circuit on the first 'yes' so worst-case latency is N
        passes for an all-safe input but typical-block latency is one.

        Per-criterion attribution is now real: the label = the exact
        rule that fired. No more "?" placeholders in per_category.
        """
        named_custom = _named_custom_criteria(custom_criteria)
        per_category: dict[str, str] = {}
        matched_label: str | None = None

        # Built-in categories first — short rule text, faster to refuse.
        for cat in builtin_categories:
            criterion = CRITERION_TEXT.get(cat)
            if not criterion:
                continue
            r = self.granite.classify_one(
                text=text, direction=direction, criterion_text=criterion
            )
            score = r["score"]
            per_category[cat] = score
            if score == "yes" and matched_label is None:
                matched_label = cat
                # Mark the rest as un-evaluated so audit logs reflect the
                # short-circuit instead of falsely claiming "no".
                for remaining in builtin_categories[
                    builtin_categories.index(cat) + 1 :
                ]:
                    per_category.setdefault(remaining, "skipped")
                for cid, _ in named_custom:
                    per_category.setdefault(cid, "skipped")
                break

        # Custom NL criteria next.
        if matched_label is None:
            for cid, ctext in named_custom:
                r = self.granite.classify_one(
                    text=text, direction=direction, criterion_text=ctext
                )
                score = r["score"]
                per_category[cid] = score
                if score == "yes":
                    matched_label = cid
                    for other_cid, _ in named_custom:
                        per_category.setdefault(other_cid, "skipped")
                    break

        return {
            "verdict": "unsafe" if matched_label else "safe",
            "label": matched_label,
            "per_category": per_category,
            "engine": "granite",
            "mode": "per_criterion",
        }

    # --- Qwen path (built-in only) ----------------------------------------

    def _classify_qwen(
        self,
        *,
        engine: QwenGuardEngine,
        text: str,
        direction: str,
        categories: list[str],
    ) -> dict[str, Any]:
        parsed = engine.classify_native(text=text, direction=direction)
        safety = parsed.get("safety")
        fired_qwen_cats = set(parsed.get("categories") or [])

        # Treat `Controversial` as `Unsafe` for blocking decisions —
        # customers are free to override the default policy.
        if safety not in ("unsafe", "controversial"):
            return {
                "verdict": "safe",
                "label": None,
                "per_category": {c: "no" for c in categories},
                "engine": engine.name,
                "mode": "qwen",
            }

        matched_label: str | None = None
        per_category: dict[str, str] = {}
        for cat in categories:
            mapped = POLICY_TO_QWEN.get(cat, frozenset())
            if mapped & fired_qwen_cats:
                per_category[cat] = "yes"
                if matched_label is None:
                    matched_label = cat
            else:
                per_category[cat] = "no"

        # Qwen flagged unsafe but the customer didn't enable any matching
        # category → don't block. The parent surfaces this in audit logs
        # but treats the request as safe per the explicit policy.
        return {
            "verdict": "unsafe" if matched_label else "safe",
            "label": matched_label,
            "per_category": per_category,
            "engine": engine.name,
            "mode": "qwen",
            "qwen_safety": safety,
            "qwen_categories": list(fired_qwen_cats),
        }


# Linux 4.14 (the kernel baked into nitro-cli's bzImage) has a virtio_vsock
# bug where recv() into buffers larger than ~16 KiB sometimes returns the
# requested length but only partially fills the buffer — leaving stale
# kernel memory in the trailing bytes. Verified by side-by-side hash
# comparison: 64 KiB+ chunks corrupt; 16 KiB chunks round-trip cleanly.
# Keep CHUNK <= 1 << 14 until the enclave kernel is updated.
RECV_CHUNK = 1 << 14


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        m = conn.recv_into(mv[got:n], n - got, socket.MSG_WAITALL)
        if m == 0:
            raise ConnectionError(f"peer closed; expected {n} bytes, got {got}")
        got += m
    return bytes(buf)


def _stream_blob(conn: socket.socket, dest: Path, expected_sha256: str, label: str) -> None:
    """Receive [8-byte BE length][bytes] from `conn`, write to `dest`, verify SHA-256."""
    n = int.from_bytes(_recv_exact(conn, 8), "big")
    log.info("receiving %s: %d bytes", label, n)
    h = hashlib.sha256()
    received = 0
    t0 = time.monotonic()
    buf = bytearray(RECV_CHUNK)
    mv = memoryview(buf)
    with dest.open("wb") as f:
        while received < n:
            want = min(RECV_CHUNK, n - received)
            m = conn.recv_into(mv[:want], want, socket.MSG_WAITALL)
            if m == 0:
                raise ConnectionError(f"peer closed mid-{label}: {received}/{n}")
            chunk = mv[:m]
            f.write(chunk)
            h.update(chunk)
            received += m
    digest = h.hexdigest()
    elapsed = time.monotonic() - t0
    log.info("received %s in %.1fs sha256=%s", label, elapsed, digest)
    if not expected_sha256:
        raise RuntimeError(
            f"refusing to load {label}: no SHA-256 baked into EIF "
            f"(set RG_{label.upper()}_SHA256 at build time)"
        )
    if digest != expected_sha256:
        raise RuntimeError(
            f"{label} digest mismatch: got {digest}, expected {expected_sha256}"
        )


def _stream_encrypted_blob(
    conn: socket.socket, dest: Path, data_key: bytes, expected_sha256: str, label: str
) -> None:
    """Receive [8-byte BE length][12-byte IV][ciphertext][16-byte tag], decrypt to `dest`."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]

    framed_len = int.from_bytes(_recv_exact(conn, 8), "big")
    if framed_len < 12 + 16:
        raise RuntimeError(f"{label}: framed len {framed_len} too small for IV+tag")
    log.info("receiving %s (encrypted): %d framed bytes", label, framed_len)

    iv = _recv_exact(conn, 12)
    payload_len = framed_len - 12 - 16

    h = hashlib.sha256()
    received = 0
    t0 = time.monotonic()
    buf = bytearray(RECV_CHUNK)
    mv = memoryview(buf)

    if not expected_sha256:
        raise RuntimeError(
            f"refusing to load {label}: no SHA-256 baked into EIF "
            f"(set RG_{label.upper()}_SHA256 at build time)"
        )

    decryptor = Cipher(algorithms.AES(data_key), modes.GCM(iv)).decryptor()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fout:
        try:
            while received < payload_len:
                want = min(RECV_CHUNK, payload_len - received)
                m = conn.recv_into(mv[:want], want, socket.MSG_WAITALL)
                if m == 0:
                    raise ConnectionError(f"peer closed mid-{label}: {received}/{payload_len}")
                pt = decryptor.update(bytes(mv[:m]))
                if pt:
                    fout.write(pt)
                    h.update(pt)
                received += m

            tag = _recv_exact(conn, 16)
            tail = decryptor.finalize_with_tag(tag)
            if tail:
                fout.write(tail)
                h.update(tail)
        except Exception:
            fout.close()
            try:
                dest.unlink()
            except OSError:
                pass
            raise

    digest = h.hexdigest()
    elapsed = time.monotonic() - t0
    log.info("decrypted %s in %.1fs sha256=%s", label, elapsed, digest)

    if digest != expected_sha256:
        try:
            dest.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"{label} plaintext digest mismatch: got {digest}, expected {expected_sha256}"
        )


def _resolve_models_to_load(header: dict[str, Any]) -> list[str]:
    """Pick which models to expect from the parent's load handshake.

    Header may pass `"models": ["granite", "qwen_06b", ...]` to choose a
    subset; absent / empty → every spec with a baked SHA-256. Unknown
    labels are rejected so a malicious manifest can't sneak in a path
    we don't expect.
    """
    requested = header.get("models")
    if isinstance(requested, list) and requested:
        labels: list[str] = []
        for entry in requested:
            label = entry["label"] if isinstance(entry, dict) else str(entry)
            if label not in MODEL_SPECS:
                raise RuntimeError(f"unknown model label in manifest: {label!r}")
            labels.append(label)
        return labels
    # Default: load everything for which we have a SHA baked in. This
    # lets the parent run "send all models" without needing to enumerate.
    return [label for label, spec in MODEL_SPECS.items() if spec.sha256]


def _load_models_from_parent() -> None:
    """Phase 1: accept ONE upload connection on LOAD_PORT, stream all configured models.

    Header JSON (parent → enclave):
        {"mode":"plaintext", "models":[{"label":"granite"}, ...]}
        {"mode":"kms",       "models":[...], "nonce_b64":"..."}

    For each label in `models` (in order): one [8-byte len][bytes] frame
    in plaintext mode, or one [8-byte framed_len][IV][ct][tag] frame in
    kms mode. The data key is negotiated once and reused for all blobs.
    """
    AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
    VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, LOAD_PORT))
    sock.listen(1)
    log.info("awaiting model upload on vsock cid=ANY port=%d", LOAD_PORT)
    conn, addr = sock.accept()
    log.info("upload connection from cid=%s port=%s", addr[0], addr[1])
    try:
        header_line = _read_line_byte(conn)
        if not header_line:
            raise ConnectionError("parent closed before sending mode header")
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bad mode header: {exc}: {header_line!r}") from exc

        mode = header.get("mode", "plaintext")
        labels = _resolve_models_to_load(header)
        log.info("load manifest: mode=%s labels=%s", mode, labels)

        data_key: bytes | None = None
        if mode == "kms":
            data_key = _negotiate_kms_data_key(conn, header)
        elif mode != "plaintext":
            raise RuntimeError(f"unknown load mode: {mode!r}")

        for label in labels:
            spec = MODEL_SPECS[label]
            dest = MODEL_DIR / spec.filename
            if mode == "kms":
                assert data_key is not None
                _stream_encrypted_blob(conn, dest, data_key, spec.sha256, label)
            else:
                _stream_blob(conn, dest, spec.sha256, label)

        conn.sendall(b'{"status":"loaded"}\n')
    finally:
        try:
            conn.close()
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def _negotiate_kms_data_key(conn: socket.socket, header: dict[str, Any]) -> bytes:
    """KMS handshake on the load connection (see module docstring)."""
    nonce = _decode_b64(header.get("nonce_b64"))
    pubkey_der = nsm.get_recipient_public_key_der()  # type: ignore[attr-defined]
    log.info("kms handshake: pubkey_der=%d bytes nonce=%s",
             len(pubkey_der), bool(nonce))
    doc = nsm.get_attestation_document(
        user_data=b"retroguard-kms-load",
        nonce=nonce,
        public_key=pubkey_der,
    )
    conn.sendall((json.dumps({"attestation_doc_b64": base64.b64encode(doc).decode()}) + "\n").encode())

    reply_line = _read_line_byte(conn)
    if not reply_line:
        raise ConnectionError("parent closed before delivering data key")
    reply = json.loads(reply_line)
    cfr_b64 = reply.get("ciphertext_for_recipient_b64")
    if not cfr_b64:
        raise RuntimeError(f"parent reply missing ciphertext_for_recipient_b64: {reply!r}")
    data_key = nsm.unwrap_kms_recipient_ciphertext(base64.b64decode(cfr_b64))
    if len(data_key) != 32:
        raise RuntimeError(f"unexpected data key length: {len(data_key)} (want 32)")
    log.info("kms handshake: unwrapped %d-byte data key", len(data_key))
    return data_key


def _decode_b64(value: str | None) -> bytes | None:
    if not value:
        return None
    return base64.b64decode(value)


def _read_line(conn: socket.socket) -> str | None:
    """Read a newline-delimited message from a stream socket.

    NB: this WILL over-read past the newline (chunks of up to 4 KiB), so
    only safe on a request/reply protocol where the rest of the buffer
    can be discarded.
    """
    chunks: list[bytes] = []
    while True:
        b = conn.recv(4096)
        if not b:
            return None
        chunks.append(b)
        if b"\n" in b:
            break
    data = b"".join(chunks)
    line, _, _ = data.partition(b"\n")
    return line.decode("utf-8", errors="replace")


def _read_line_byte(conn: socket.socket) -> str | None:
    """Read a newline-delimited message one byte at a time."""
    out = bytearray()
    while True:
        b = conn.recv(1)
        if not b:
            return None
        if b == b"\n":
            break
        out.extend(b)
    return out.decode("utf-8", errors="replace")


_PROVIDER_ROUTING: dict[str, dict[str, Any]] = {
    "openai": {
        "host": "api.openai.com",
        "path": "/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_format": "Bearer {key}",
        "extra_headers": {},
    },
    "anthropic": {
        "host": "api.anthropic.com",
        "path": "/v1/messages",
        "auth_header": "x-api-key",
        "auth_format": "{key}",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
}


def _send_frame(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _handle_chat(conn: socket.socket, msg: dict[str, Any]) -> None:
    log.info("op:chat received request_id=%s provider=%s stream=%s",
             msg.get("request_id"), msg.get("provider"), msg.get("stream"))
    request_id = msg.get("request_id", "")
    provider = msg.get("provider", "")
    body = msg.get("body") or {}
    is_stream = bool(msg.get("stream", False))
    recipient_ct_b64 = msg.get("recipient_ciphertext_b64") or ""

    routing = _PROVIDER_ROUTING.get(provider)
    if routing is None:
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"unknown provider: {provider}"})
        return
    if not recipient_ct_b64:
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": "missing recipient_ciphertext_b64"})
        return

    try:
        recipient_ct = base64.b64decode(recipient_ct_b64)
        log.info("op:chat unwrapping recipient_ct len=%d", len(recipient_ct))
        api_key = nsm.unwrap_kms_recipient_ciphertext(recipient_ct).decode("utf-8")
        log.info("op:chat unwrap ok api_key_prefix=%s", api_key[:6] + "***")
    except Exception as exc:
        log.exception("op:chat recipient_unwrap_failed: %s", exc)
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"recipient_unwrap_failed: {exc}"})
        return

    upstream_host = routing["host"]
    vsock_port = DEFAULT_UPSTREAM_PORTS.get(upstream_host)
    if vsock_port is None:
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"no vsock_port for {upstream_host}"})
        return
    try:
        log.info("op:chat opening TLS over vsock host=%s vsock_port=%d", upstream_host, vsock_port)
        tls = open_tls_over_vsock(upstream_host=upstream_host, vsock_port=vsock_port)
        log.info("op:chat TLS handshake ok")
    except Exception as exc:
        log.exception("op:chat vsock_tls_open_failed")
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"vsock_tls_open_failed: {exc}"})
        return

    try:
        request_body = {**body, "stream": is_stream}
        body_bytes = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if is_stream else "application/json",
            routing["auth_header"]: routing["auth_format"].format(key=api_key),
            **routing["extra_headers"],
        }
        log.info("op:chat sending POST %s body_len=%d", routing["path"], len(body_bytes))
        send_request(
            tls,
            method="POST",
            path=routing["path"],
            host=upstream_host,
            headers=headers,
            body=body_bytes,
        )

        api_key = ""

        reader = HttpReader(tls)
        try:
            log.info("op:chat reading response status")
            status = reader.read_status()
            log.info("op:chat upstream status=%d", status)
            response_headers = reader.read_headers()
            log.info("op:chat got %d response headers", len(response_headers))
        except HttpError as exc:
            _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"upstream_head_read_failed: {exc}"})
            return

        is_chunked = response_headers.get("transfer-encoding", "").lower() == "chunked"
        content_length_str = response_headers.get("content-length")

        if is_stream:
            _send_frame(conn, {"event": "start", "request_id": request_id, "status": status})
            try:
                if is_chunked:
                    leftover = b""
                    for chunk in reader.iter_chunked():
                        leftover += chunk
                        while b"\n\n" in leftover:
                            event, _, leftover = leftover.partition(b"\n\n")
                            text = event.decode("utf-8", errors="replace")
                            for line in text.splitlines():
                                if line:
                                    _send_frame(conn, {"event": "chunk", "request_id": request_id, "sse_line": line})
                    if leftover:
                        text = leftover.decode("utf-8", errors="replace")
                        for line in text.splitlines():
                            if line:
                                _send_frame(conn, {"event": "chunk", "request_id": request_id, "sse_line": line})
                else:
                    raw = reader.read_until_close()
                    text = raw.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        if line:
                            _send_frame(conn, {"event": "chunk", "request_id": request_id, "sse_line": line})
            except HttpError as exc:
                _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"upstream_body_read_failed: {exc}", "upstream_status": status})
                return
            _send_frame(conn, {"event": "done", "request_id": request_id})
        else:
            try:
                if is_chunked:
                    body_buf = bytearray()
                    for chunk in reader.iter_chunked():
                        body_buf.extend(chunk)
                    raw = bytes(body_buf)
                elif content_length_str is not None:
                    raw = reader.read_fixed(int(content_length_str))
                else:
                    raw = reader.read_until_close()
            except (HttpError, ValueError) as exc:
                _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"upstream_body_read_failed: {exc}", "upstream_status": status})
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"upstream_body_parse_failed: {exc}", "upstream_status": status})
                return
            _send_frame(conn, {"event": "buffered", "request_id": request_id, "status": status, "body": payload})
    finally:
        try:
            tls.close()
        except OSError:
            pass


def _serve_one(conn: socket.socket, classifier: TieredClassifier) -> None:
    try:
        while True:
            raw = _read_line(conn)
            if raw is None:
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                conn.sendall((json.dumps({"error": f"bad_json: {e}"}) + "\n").encode())
                continue

            op = msg.get("op")
            if op == "attest":
                t0 = time.monotonic()
                user_data = _decode_b64(msg.get("user_data_b64"))
                nonce = _decode_b64(msg.get("nonce_b64"))
                if msg.get("embed_recipient_pubkey"):
                    public_key = nsm.get_recipient_public_key_der()
                else:
                    public_key = _decode_b64(msg.get("public_key_b64"))
                try:
                    doc = nsm.get_attestation_document(
                        user_data=user_data, nonce=nonce, public_key=public_key
                    )
                    payload = {
                        "request_id": msg.get("request_id"),
                        "attestation_doc_b64": base64.b64encode(doc).decode(),
                        "latency_ms": round((time.monotonic() - t0) * 1000, 2),
                    }
                except Exception as exc:
                    payload = {
                        "request_id": msg.get("request_id"),
                        "error": f"nsm_attest_failed: {exc}",
                    }
                conn.sendall((json.dumps(payload) + "\n").encode())
                continue

            if op == "chat":
                _handle_chat(conn, msg)
                continue

            if op != "classify":
                conn.sendall((json.dumps({"error": "unknown_op"}) + "\n").encode())
                continue

            t0 = time.monotonic()
            categories = msg.get("categories")
            custom_criteria = msg.get("custom_criteria") or []
            has_custom = any((c.get("text") or "").strip() for c in custom_criteria)

            if (not isinstance(categories, list) or not categories) and not has_custom:
                # No categories AND no custom criteria → nothing to evaluate.
                conn.sendall(
                    (
                        json.dumps(
                            {
                                "request_id": msg.get("request_id"),
                                "verdict": "safe",
                                "label": None,
                                "per_category": {},
                                "engine": "noop",
                                "latency_ms": 0.0,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                continue

            tier_raw = str(msg.get("protection_effort") or "expert").lower()
            tier = tier_raw if tier_raw in ("fast", "expert", "heavy") else "expert"
            result = classifier.classify(
                text=msg.get("text", ""),
                direction=msg.get("direction", "input"),
                categories=[str(c) for c in (categories or [])],
                protection_effort=tier,
                custom_criteria=custom_criteria,
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            payload = {
                "request_id": msg.get("request_id"),
                "verdict": result["verdict"],
                "label": result["label"],
                "per_category": result["per_category"],
                "engine": result.get("engine"),
                "mode": result.get("mode"),
                "latency_ms": latency_ms,
            }
            conn.sendall((json.dumps(payload) + "\n").encode())
    except Exception as e:
        log.exception("connection error: %s", e)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _build_tiered_classifier() -> TieredClassifier:
    """Eager-load all four engines. Boot order: small to large so the
    biggest mmap allocation happens last (less fragmentation pressure)."""
    granite = GraniteGuardEngine(model_path=_model_path("granite"))
    qwen_06b = QwenGuardEngine(name="qwen_06b", model_path=_model_path("qwen_06b"))
    qwen_4b = QwenGuardEngine(name="qwen_4b", model_path=_model_path("qwen_4b"))
    qwen_8b = QwenGuardEngine(name="qwen_8b", model_path=_model_path("qwen_8b"))
    return TieredClassifier(
        granite=granite, qwen_06b=qwen_06b, qwen_4b=qwen_4b, qwen_8b=qwen_8b,
    )


def main() -> int:
    import threading

    log.info("python=%s exe=%s", sys.version.split()[0], sys.executable)
    log.info("sys.path[:5]=%s", sys.path[:5])
    _load_models_from_parent()
    classifier = _build_tiered_classifier()

    AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
    VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)

    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, CLASSIFY_PORT))
    sock.listen(64)
    log.info("listening on vsock cid=ANY port=%d", CLASSIFY_PORT)

    while True:
        conn, addr = sock.accept()
        log.info("accepted from cid=%s port=%s", addr[0], addr[1])
        threading.Thread(
            target=_serve_one, args=(conn, classifier), daemon=True
        ).start()


if __name__ == "__main__":
    sys.exit(main())

"""Tests for the load-handshake manifest resolution.

`_resolve_models_to_load` decides which model labels the enclave will
expect (in order) on the upload connection. We test the three header
shapes that matter:
  - explicit list of label strings
  - explicit list of {label: ...} dicts
  - missing/empty → all SHA-baked specs
And reject unknown labels (defense against a malicious manifest).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _bake_shas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend all four SHAs are baked. Without this the default
    fallback list is empty and the omit-models test asserts emptiness."""
    monkeypatch.setenv("RG_GRANITE_SHA256", "a" * 64)
    monkeypatch.setenv("RG_QWEN_06B_SHA256", "b" * 64)
    monkeypatch.setenv("RG_QWEN_4B_SHA256", "c" * 64)
    monkeypatch.setenv("RG_QWEN_8B_SHA256", "d" * 64)
    # Reload the module so MODEL_SPECS picks up the env at import time.
    if "retroguard_classifier.server" in sys.modules:
        del sys.modules["retroguard_classifier.server"]


class TestResolveModelsToLoad:
    def test_explicit_string_list(self) -> None:
        from retroguard_classifier.server import _resolve_models_to_load
        labels = _resolve_models_to_load({"models": ["granite", "qwen_8b"]})
        assert labels == ["granite", "qwen_8b"]

    def test_explicit_dict_list(self) -> None:
        from retroguard_classifier.server import _resolve_models_to_load
        labels = _resolve_models_to_load(
            {"models": [{"label": "qwen_06b"}, {"label": "qwen_4b"}]}
        )
        assert labels == ["qwen_06b", "qwen_4b"]

    def test_missing_models_returns_all_with_baked_sha(self) -> None:
        from retroguard_classifier.server import _resolve_models_to_load
        labels = _resolve_models_to_load({})
        assert set(labels) == {"granite", "qwen_06b", "qwen_4b", "qwen_8b"}

    def test_unknown_label_rejected(self) -> None:
        from retroguard_classifier.server import _resolve_models_to_load
        with pytest.raises(RuntimeError, match="unknown model label"):
            _resolve_models_to_load({"models": ["nope"]})


class TestModelSpecPathing:
    def test_each_label_has_distinct_filename(self) -> None:
        from retroguard_classifier.server import MODEL_SPECS
        filenames = {spec.filename for spec in MODEL_SPECS.values()}
        assert len(filenames) == len(MODEL_SPECS)

    def test_specs_pick_up_env_shas(self) -> None:
        from retroguard_classifier.server import MODEL_SPECS
        assert MODEL_SPECS["granite"].sha256 == "a" * 64
        assert MODEL_SPECS["qwen_06b"].sha256 == "b" * 64
        assert MODEL_SPECS["qwen_4b"].sha256 == "c" * 64
        assert MODEL_SPECS["qwen_8b"].sha256 == "d" * 64

    def test_path_resolves_under_model_dir(self) -> None:
        from retroguard_classifier.server import MODEL_DIR, _model_path
        for label in ("granite", "qwen_06b", "qwen_4b", "qwen_8b"):
            assert _model_path(label).parent == MODEL_DIR

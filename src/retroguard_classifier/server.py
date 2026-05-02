"""In-enclave classifier server.

Runs inside the Nitro Enclave. The 8B Q4_K_M dual-model bundle is
~10GB — too large to bake into the EIF cpio (nitro-cli 1.4.4 caps
useful initramfs around ~5GB). Instead, the parent streams the
GGUF blobs over vsock at boot and the enclave verifies each blob's
SHA-256 against constants baked into the image (and therefore
measured by PCR0).

Trust model:
  * The parent CAN read the model bytes (Granite Guardian / Qwen3Guard
    are public). What it cannot do is substitute a different file: the
    enclave refuses any blob whose digest does not match the baked
    constants, and any tampering with those constants changes PCR0.
  * Customers attest the EIF (PCRs), then verify the constants are the
    expected mradermacher Q4_K_M digests against the transparency log.

Wire protocol:
  * Port 5006 (load): one connection. Two framed blobs back-to-back.
        [8-byte BE length][bytes]   ← Granite GGUF
        [8-byte BE length][bytes]   ← Qwen3Guard GGUF
    Connection closes after both verified. If verification fails the
    enclave logs and exits.
  * Port 5005 (classify): newline-delimited JSON.
        Request:  {"op":"classify","request_id":"...","text":"...","direction":"input|output"}
        Response: {"request_id":"...","verdict":"safe|unsafe","label":"...","latency_ms":...}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enclave")

LOAD_PORT = int(os.environ.get("RG_LOAD_PORT", "5006"))
CLASSIFY_PORT = int(os.environ.get("RG_VSOCK_PORT", "5005"))
MODEL_DIR = Path(os.environ.get("RG_MODEL_DIR", "/tmp/models"))
GRANITE_PATH = MODEL_DIR / "granite-guardian.gguf"
QWEN_PATH = MODEL_DIR / "qwen3guard.gguf"
GRANITE_SHA256 = os.environ.get("RG_GRANITE_SHA256", "").lower()
QWEN_SHA256 = os.environ.get("RG_QWEN_SHA256", "").lower()
MODEL_CTX = int(os.environ.get("RG_MODEL_CTX", "4096"))
MODEL_THREADS = int(os.environ.get("RG_MODEL_THREADS", "32"))


def _build_granite_prompt(text: str, direction: str) -> str:
    """Granite Guardian chat template.

    Granite Guardian uses chat-style turns with system + user/assistant.
    It returns 'Yes' (unsafe) or 'No' (safe) on the first line. Optionally
    a category list on the second line.
    """
    role = "user" if direction == "input" else "assistant"
    return (
        "<|start_of_role|>system<|end_of_role|>You are a safety classifier. "
        "Evaluate the following message for harmful content (categories: "
        "harm, social_bias, jailbreaking, violence, profanity, sexual_content, "
        "unethical_behaviour). On the first line, respond with exactly 'Yes' "
        "if unsafe or 'No' if safe. If 'Yes', on the second line, list "
        "violated category names comma-separated.<|end_of_text|>\n"
        f"<|start_of_role|>{role}<|end_of_role|>{text}<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )


def _build_qwen3guard_prompt(text: str, direction: str) -> str:
    """Qwen3Guard-Gen chat template (ChatML)."""
    role = "user" if direction == "input" else "assistant"
    return (
        "<|im_start|>system\n"
        "You are a safety guardrail. Classify the following message for harmful "
        "content. Respond with 'safe' or 'unsafe' on the first line. If unsafe, "
        "list violated categories on the second line.\n<|im_end|>\n"
        f"<|im_start|>{role}\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class GuardrailEngine:
    """Wraps a single GGUF model loaded via llama-cpp-python."""

    def __init__(self, name: str, model_path: Path, prompt_builder, stop_tokens: list[str]) -> None:
        log.info("loading %s from %s (ctx=%d, threads=%d)", name, model_path, MODEL_CTX, MODEL_THREADS)
        from llama_cpp import Llama  # type: ignore[import-not-found]

        self._name = name
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=MODEL_CTX,
            n_threads=MODEL_THREADS,
            n_batch=512,
            verbose=False,
        )
        self._prompt_builder = prompt_builder
        self._stop_tokens = stop_tokens
        log.info("loaded %s", name)

    def classify(self, text: str, direction: str) -> dict[str, Any]:
        prompt = self._prompt_builder(text, direction)
        out = self._llm(
            prompt,
            max_tokens=32,
            temperature=0.0,
            stop=self._stop_tokens,
        )
        raw = (out.get("choices", [{}])[0].get("text") or "").strip()
        first_line = raw.splitlines()[0].strip().lower() if raw else ""
        unsafe = first_line.startswith("yes") or first_line.startswith("unsafe")
        verdict = "unsafe" if unsafe else "safe"
        label: str | None = None
        if verdict == "unsafe" and len(raw.splitlines()) > 1:
            label = raw.splitlines()[1].split(",")[0].strip().lower() or None
        return {"verdict": verdict, "label": label, "engine": self._name, "raw": raw[:200]}


class DualClassifier:
    """Routes to Granite (input) or Qwen3Guard (output) per spec §3.

    If only Granite is loaded (Qwen path missing), falls back to Granite
    for both directions.
    """

    def __init__(self) -> None:
        self.granite = GuardrailEngine(
            "granite-guardian", GRANITE_PATH, _build_granite_prompt,
            stop_tokens=["<|end_of_text|>", "</s>"],
        )
        self.qwen: GuardrailEngine | None = None
        if QWEN_PATH.exists():
            self.qwen = GuardrailEngine(
                "qwen3guard-gen", QWEN_PATH, _build_qwen3guard_prompt,
                stop_tokens=["<|im_end|>", "</s>"],
            )
            log.info("dual-classifier mode: granite + qwen3guard")
        else:
            log.info("single-classifier mode: granite only (qwen path missing)")

    def classify(self, text: str, direction: str) -> dict[str, Any]:
        engine = self.granite if (direction == "input" or self.qwen is None) else self.qwen
        return engine.classify(text, direction)


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
    """Receive [8-byte BE length][bytes] from `conn`, write to `dest`, verify SHA-256.

    Streams to disk in RECV_CHUNK-sized blocks (see comment on the constant
    for the vsock kernel-bug rationale). Raises on hash mismatch — caller
    exits the enclave so a tampered upload cannot reach the classifier.
    """
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


def _load_models_from_parent() -> None:
    """Phase 1: accept ONE upload connection on LOAD_PORT, stream both models.

    Blocks until both files are written and verified. Subsequent
    uploads are ignored (the load port is closed after success).
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
        _stream_blob(conn, GRANITE_PATH, GRANITE_SHA256, "granite")
        _stream_blob(conn, QWEN_PATH, QWEN_SHA256, "qwen")
        # Acknowledge so the parent knows both blobs were verified.
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


def _read_line(conn: socket.socket) -> str | None:
    """Read a newline-delimited message from a stream socket."""
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


def _serve_one(conn: socket.socket, classifier: DualClassifier) -> None:
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

            if msg.get("op") != "classify":
                conn.sendall((json.dumps({"error": "unknown_op"}) + "\n").encode())
                continue

            t0 = time.monotonic()
            result = classifier.classify(
                text=msg.get("text", ""),
                direction=msg.get("direction", "input"),
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            payload = {
                "request_id": msg.get("request_id"),
                "verdict": result["verdict"],
                "label": result["label"],
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


def main() -> int:
    log.info("python=%s exe=%s", sys.version.split()[0], sys.executable)
    log.info("sys.path[:5]=%s", sys.path[:5])
    _load_models_from_parent()
    classifier = DualClassifier()

    AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
    VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)

    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, CLASSIFY_PORT))
    sock.listen(8)
    log.info("listening on vsock cid=ANY port=%d", CLASSIFY_PORT)

    while True:
        conn, addr = sock.accept()
        log.info("accepted from cid=%s port=%s", addr[0], addr[1])
        _serve_one(conn, classifier)


if __name__ == "__main__":
    sys.exit(main())

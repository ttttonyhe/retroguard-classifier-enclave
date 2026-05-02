"""In-enclave classifier server.

Runs inside the Nitro Enclave. Listens on vsock and classifies arbitrary
text against the configured guardrail categories using llama.cpp +
Granite Guardian (loaded once at startup).

Wire protocol (newline-delimited JSON over a single vsock connection):

    Request (parent → enclave):
        {"op": "classify", "request_id": "uuid", "text": "...", "direction": "input|output"}

    Response (enclave → parent):
        {"request_id": "uuid", "verdict": "safe|unsafe", "label": "harm|...", "score": 0.0..1.0}

Notes:
  * Single connection serialized for v1; batching/concurrency lands later.
  * Granite Guardian classifier prompt template: see prompts.py.
  * Model path: /opt/models/granite-guardian.gguf (baked into EIF for v1;
    KMS-attested decrypt at runtime in v2).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enclave")

VSOCK_PORT = int(os.environ.get("RG_VSOCK_PORT", "5005"))
MODEL_PATH = Path(os.environ.get("RG_MODEL_PATH", "/opt/models/granite-guardian.gguf"))
MODEL_CTX = int(os.environ.get("RG_MODEL_CTX", "4096"))
MODEL_THREADS = int(os.environ.get("RG_MODEL_THREADS", "32"))


def _build_classifier_prompt(text: str, direction: str) -> str:
    """Granite Guardian classifier prompt template.

    Granite Guardian expects a chat-style prompt with the user message
    being the content under evaluation; the response is `safe`/`unsafe`
    with optional category labels.
    """
    role = "user" if direction == "input" else "assistant"
    return (
        "You are a safety classifier. Evaluate the following message for harmful "
        "content (categories: harm, jailbreaking, violence, social_bias). Respond "
        "with exactly one word: `safe` or `unsafe`.\n\n"
        f"<{role}>\n{text}\n</{role}>\n\n"
        "Verdict:"
    )


class GraniteGuardianClassifier:
    def __init__(self, model_path: Path) -> None:
        log.info("loading model %s (ctx=%d, threads=%d)", model_path, MODEL_CTX, MODEL_THREADS)
        # Lazy-import llama_cpp so import errors don't kill the server before
        # we can log them. v1 Pin: use llama-cpp-python.
        from llama_cpp import Llama  # type: ignore[import-not-found]

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=MODEL_CTX,
            n_threads=MODEL_THREADS,
            n_batch=512,
            verbose=False,
        )
        log.info("model loaded")

    def classify(self, text: str, direction: str) -> dict[str, Any]:
        prompt = _build_classifier_prompt(text, direction)
        out = self._llm(
            prompt,
            max_tokens=8,
            temperature=0.0,
            stop=["\n", "."],
        )
        verdict_raw = (out.get("choices", [{}])[0].get("text") or "").strip().lower()
        verdict = "unsafe" if "unsafe" in verdict_raw else "safe"
        # v1 doesn't extract per-category labels yet; that lands when we
        # wire the structured-output template per spec §6.
        return {"verdict": verdict, "label": "general" if verdict == "unsafe" else None}


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


def _serve_one(conn: socket.socket, classifier: GraniteGuardianClassifier) -> None:
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
    if not MODEL_PATH.exists():
        log.error("model not found: %s", MODEL_PATH)
        return 2

    classifier = GraniteGuardianClassifier(MODEL_PATH)

    # AF_VSOCK = 40 (Linux). Listen on (CID_ANY, port) — accept any incoming.
    AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
    VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)

    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(8)
    log.info("listening on vsock cid=ANY port=%d", VSOCK_PORT)

    # Single-connection serialized for v1 (model is single-instance).
    while True:
        conn, addr = sock.accept()
        log.info("accepted from cid=%s port=%s", addr[0], addr[1])
        # Serve in-thread but synchronously — no concurrent classify.
        # Concurrent batching lands when we wire llama.cpp's batch-decode.
        _serve_one(conn, classifier)


if __name__ == "__main__":
    sys.exit(main())

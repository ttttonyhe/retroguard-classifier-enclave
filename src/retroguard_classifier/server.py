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
        Request:  {"op":"classify","request_id":"...","text":"...",
                   "direction":"input|output","categories":["harm",...]}
        Response: {"request_id":"...","verdict":"safe|unsafe","label":"<cat>"|null,
                   "per_category":{"<cat>":"yes|no",...},"latency_ms":...}
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
GRANITE_PATH = MODEL_DIR / "granite-guardian.gguf"
GRANITE_SHA256 = os.environ.get("RG_GRANITE_SHA256", "").lower()
MODEL_CTX = int(os.environ.get("RG_MODEL_CTX", "4096"))
MODEL_THREADS = int(os.environ.get("RG_MODEL_THREADS", "32"))


# --------------------------------------------------------------------------
# Official Granite Guardian 4.1 prompt template per IBM model card.
# https://huggingface.co/ibm-granite/granite-guardian-4.1-8b
#
# We use no-think mode (faster — emits empty <think> tags then the
# verdict) for the warm-path serving loop. Output is always:
#     <think>\n</think>\n<score>yes|no</score>
# Parsed with `_parse_score` below. The `</think>` we kept seeing as a
# "label" in earlier attempts wasn't a model bug — it was the close-tag
# of the structured envelope we never knew to expect.
# --------------------------------------------------------------------------

GUARDIAN_INSTRUCTION_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)

# Pre-baked criterion text — verbatim phrasings that map onto IBM's
# six trained safety categories. Granite was trained on each criterion
# independently; per-category classification gives the strongest signal.
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


def _build_guardian_block(criterion_text: str, scoring_target: str) -> str:
    """Assemble the IBM-spec'd <guardian> message body.

    `scoring_target` lets us swap the schema language between user-text
    checks (input direction) and assistant-text checks (output direction)
    while keeping the structured shape Granite was trained on.
    """
    return (
        f"<guardian>{GUARDIAN_INSTRUCTION_NOTHINK}\n\n"
        f"### Criteria: {criterion_text}\n\n"
        f"### Scoring Schema: If {scoring_target} meets the criteria, "
        "return 'yes'; otherwise, return 'no'."
    )


def _build_granite_judge_prompt(text: str, direction: str, criterion_text: str) -> str:
    """Build the official Granite chat-template-formatted prompt.

    For input checks we score the user message directly; for output
    checks we present the user/assistant turn pair so the judge has
    full conversation context (Granite was trained that way per the
    "last assistant's text" schema).
    """
    if direction == "input":
        guardian = _build_guardian_block(criterion_text, "the user's message")
        return (
            f"<|start_of_role|>user<|end_of_role|>{text}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{guardian}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )
    # output: the prior user turn isn't visible to the classifier, so
    # we synthesize a placeholder that doesn't bias the judgement.
    guardian = _build_guardian_block(criterion_text, "the last assistant's text")
    return (
        "<|start_of_role|>user<|end_of_role|>(prior user turn)<|end_of_text|>\n"
        f"<|start_of_role|>assistant<|end_of_role|>{text}<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{guardian}<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )


# `</score>` is in the stop-token set so llama-cpp halts generation as
# soon as Granite tries to close the tag (saves ~10 tokens of latency).
# That means the raw text we see ends with `<score> yes` or `<score> no`
# — the closing tag was eaten by the stop matcher. Match either form.
_SCORE_RE = re.compile(
    r"<score>\s*(yes|no)\b",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_score(raw: str) -> str | None:
    """Pull the verdict out of `<think>...</think><score>yes|no(</score>)?`.

    Returns "yes", "no", or None when the model drifted off-template.
    """
    cleaned = _THINK_RE.sub("", raw or "", count=1).strip()
    m = _SCORE_RE.search(cleaned)
    if not m:
        return None
    return m.group(1).lower()


class GuardrailEngine:
    """Wraps the Granite Guardian 4.1 GGUF loaded via llama-cpp-python.

    Each `classify` call evaluates the text against ONE criterion. The
    DualClassifier wraps this with per-category dispatch — a 3-category
    policy (jailbreaking + violence + harm) costs 3 forward passes;
    Granite was trained per-criterion, so this is the IBM-recommended
    accuracy/latency tradeoff.
    """

    # Stop on `</score>` so we close out as soon as the verdict lands.
    # No-think mode emits roughly:
    #   `<think>\n</think>\n<score>yes`  (close tag consumed by stop)
    # which is ~10 tokens — `MAX_TOKENS=16` gives a small safety margin
    # without paying for runaway generation.
    STOP_TOKENS = ["</score>", "<|end_of_text|>", "</s>"]
    MAX_TOKENS = 16

    def __init__(self, name: str, model_path: Path) -> None:
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
        log.info("loaded %s", name)

    def classify_one(self, *, text: str, direction: str, criterion_text: str) -> dict[str, Any]:
        """Evaluate `text` against a single criterion. Returns a yes/no score."""
        prompt = _build_granite_judge_prompt(text, direction, criterion_text)
        out = self._llm(
            prompt,
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            stop=self.STOP_TOKENS,
        )
        raw = (out.get("choices", [{}])[0].get("text") or "").strip()
        score = _parse_score(raw)
        # Failure-to-parse is treated as `no` (safe) per the model card's
        # binary contract — Granite's verdict tag was missing or garbled,
        # which historically meant the request didn't even score (e.g.
        # truncated mid-`<think>`). Logged so we can audit accuracy.
        if score is None:
            log.warning(
                "granite verdict unparseable; treating as 'no' (safe). raw=%r", raw[:200]
            )
        return {"score": score or "no", "raw": raw[:200]}


def _build_combined_criterion(categories: list[str]) -> str:
    """One BYOC criterion that ORs together every enabled category.

    Granite was trained per-category, but the model card explicitly
    supports custom natural-language criteria. Combining lets us answer
    the policy question ("is this text in violation of any enabled
    category?") in a single forward pass — ~3-5x faster than checking
    each category sequentially. The trade-off: when blocked, we know
    *something* in the policy fired but can't pin which one without
    a second pass.
    """
    bullets = []
    for cat in categories:
        text = CRITERION_TEXT.get(cat)
        if text:
            bullets.append(f"  - {text}")
    body = "\n".join(bullets)
    return (
        "The text violates the safety policy if any of the following criteria apply:\n"
        f"{body}"
    )


class DualClassifier:
    """Granite-Guardian dispatcher with two latency profiles.

    `combined=True` (default, fast):
        ONE forward pass with a BYOC criterion that ORs together every
        enabled category. ~4s on c7i CPU regardless of category count.
        On unsafe verdict, label = first enabled category (we lose the
        per-category attribution but gain ~3x latency).

    `combined=False` (per-category, more attribution):
        Sequential pass per category, short-circuiting on first match.
        ~4s × N categories worst case. Useful for telemetry / shadow
        evaluation runs where attribution matters.

    Wire contract: `(text, direction, categories, combined?)` — the
    parent passes the policy's `enabled_categories` plus an optional
    `combined` flag. Empty `categories` skips classification entirely.

    Returns:
        {
          "verdict": "safe" | "unsafe",
          "label": <category> | None,
          "per_category": {<cat>: "yes"|"no"|"skipped"},
          "mode": "combined" | "per_category",
        }
    """

    def __init__(self) -> None:
        self.engine = GuardrailEngine("safety-classifier", GRANITE_PATH)
        log.info("safety classifier loaded (single-engine)")

    def classify(
        self,
        *,
        text: str,
        direction: str,
        categories: list[str],
        combined: bool = True,
    ) -> dict[str, Any]:
        valid = [c for c in categories if c in CRITERION_TEXT]
        unknown = [c for c in categories if c not in CRITERION_TEXT]
        for c in unknown:
            log.warning("unknown criterion category: %r — skipping", c)

        if not valid:
            return {"verdict": "safe", "label": None, "per_category": {}, "mode": "noop"}

        if combined:
            criterion = _build_combined_criterion(valid)
            r = self.engine.classify_one(
                text=text, direction=direction, criterion_text=criterion
            )
            unsafe = r["score"] == "yes"
            return {
                "verdict": "unsafe" if unsafe else "safe",
                # We can't attribute which category fired on a combined
                # check. Surface the first one as the label so the
                # customer-visible code is at least in their policy.
                "label": valid[0] if unsafe else None,
                "per_category": {cat: ("?" if unsafe else "no") for cat in valid},
                "mode": "combined",
            }

        # Per-category, short-circuit on first match.
        per_category: dict[str, str] = {}
        first_match: str | None = None
        for cat in valid:
            criterion = CRITERION_TEXT[cat]
            r = self.engine.classify_one(
                text=text, direction=direction, criterion_text=criterion
            )
            per_category[cat] = r["score"]
            if r["score"] == "yes" and first_match is None:
                first_match = cat
                break
        # Mark un-evaluated categories as "skipped" rather than "no"
        # so audit logs don't read like we cleared everything.
        for cat in valid:
            per_category.setdefault(cat, "skipped")
        return {
            "verdict": "unsafe" if first_match else "safe",
            "label": first_match,
            "per_category": per_category,
            "mode": "per_category",
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


def _stream_encrypted_blob(
    conn: socket.socket, dest: Path, data_key: bytes, expected_sha256: str, label: str
) -> None:
    """Receive [8-byte BE length][12-byte IV][ciphertext][16-byte tag], decrypt to `dest`.

    AES-256-GCM with the provided data_key. We stream-decrypt directly
    to disk + a hashing context — never materializing the full 5 GiB
    plaintext in memory. The GCM tag is verified at the end via
    `finalize_with_tag`; if it fails, the partial file is removed.
    """
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
            # Partial plaintext on disk could be mistaken for a valid
            # model on a subsequent boot — wipe it on any failure.
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


def _load_models_from_parent() -> None:
    """Phase 1: accept ONE upload connection on LOAD_PORT, stream both models.

    Two modes selected by the parent's first newline-delimited JSON message:

      * `{"mode":"plaintext"}` (default) — stream cleartext GGUFs; we hash
        each on the way in and compare against PCR-baked constants.
      * `{"mode":"kms"}` — KMS-attested decrypt path:
          1. We reply with `{"attestation_doc_b64":...}` (the doc embeds
             our per-enclave RSA pubkey).
          2. Parent calls `kms.Decrypt(Recipient=...)` and sends back
             `{"ciphertext_for_recipient_b64":...}`.
          3. We RSA-OAEP-SHA256 unwrap to a 32-byte AES-256 data key.
          4. Parent then streams [iv][ciphertext][tag]-framed blobs and
             we AES-GCM-decrypt + SHA-verify the plaintexts.

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
        # IMPORTANT: read the JSON header byte-by-byte. _read_line() buffers
        # in 4 KiB chunks and silently drops anything past the newline — that
        # would eat the 8-byte length prefix (and the start of the GGUF) and
        # the next read would interpret model bytes as a length, hanging on
        # an exabyte-sized recv. The header is small (< 200 B), so the
        # per-byte recv cost is irrelevant.
        header_line = _read_line_byte(conn)
        if not header_line:
            raise ConnectionError("parent closed before sending mode header")
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bad mode header: {exc}: {header_line!r}") from exc

        mode = header.get("mode", "plaintext")
        if mode == "plaintext":
            _stream_blob(conn, GRANITE_PATH, GRANITE_SHA256, "granite")
        elif mode == "kms":
            data_key = _negotiate_kms_data_key(conn, header)
            _stream_encrypted_blob(conn, GRANITE_PATH, data_key, GRANITE_SHA256, "granite")
        else:
            raise RuntimeError(f"unknown load mode: {mode!r}")

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
    """KMS handshake on the load connection.

    1. Generate an attestation document binding the enclave's recipient
       public key + the parent-supplied nonce (if any).
    2. Send `{"attestation_doc_b64":...}` so the parent can pass it as
       `Recipient.AttestationDocument` to `kms.Decrypt`.
    3. Read `{"ciphertext_for_recipient_b64":...}` and unwrap the
       returned 32-byte data key with the matching private key.
    """
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
    """Tolerantly decode an optional base64 field from a vsock message."""
    if not value:
        return None
    return base64.b64decode(value)


def _read_line(conn: socket.socket) -> str | None:
    """Read a newline-delimited message from a stream socket.

    NB: this WILL over-read past the newline (chunks of up to 4 KiB), so
    only safe on a request/reply protocol where the rest of the buffer
    can be discarded. For the `_load_models_from_parent` handshake (where
    a length-prefixed binary blob follows the JSON header on the same
    connection) use _read_line_byte() instead.
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
    """Read a newline-delimited message one byte at a time.

    Slow but correct: leaves any subsequent bytes on the socket so a
    follow-up _recv_exact() / recv_into() reads them as expected. Use
    this whenever the next message on the wire is a binary blob.
    """
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
    """Proxy a chat completion through to the upstream provider.

    Wire (parent → enclave):
        {"op":"chat","request_id":...,"recipient_ciphertext_b64":...,
         "provider":"openai|anthropic","body":{...},"stream":bool}

    Wire (enclave → parent), one or more newline-delimited frames:
        Buffered: {"event":"buffered","status":int,"body":{...}}
        Streaming: {"event":"start","status":int}
                   {"event":"chunk","sse_line":"data: ..."}
                   ... (repeated)
                   {"event":"done"}
        Errors:   {"event":"error","message":str,"upstream_status":int|None}
    """
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

    # 1. Recover the customer's plaintext API key. Lives in this stack frame
    # only; we don't log it, don't forward it back to the parent.
    try:
        recipient_ct = base64.b64decode(recipient_ct_b64)
        log.info("op:chat unwrapping recipient_ct len=%d", len(recipient_ct))
        api_key = nsm.unwrap_kms_recipient_ciphertext(recipient_ct).decode("utf-8")
        log.info("op:chat unwrap ok api_key_prefix=%s", api_key[:6] + "***")
    except Exception as exc:
        log.exception("op:chat recipient_unwrap_failed: %s", exc)
        _send_frame(conn, {"event": "error", "request_id": request_id, "message": f"recipient_unwrap_failed: {exc}"})
        return

    # 2. Open a TLS-over-vsock socket to the upstream via the parent's vsock-proxy.
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
        # 3. Build + send the upstream request. Force `stream` to match what
        # the parent asked for so the response framing is consistent.
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

        # Burn the api_key local; it's been written to the TLS socket already.
        api_key = ""

        # 4. Parse response head.
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

        # 5. Stream or buffer body back through vsock.
        is_chunked = response_headers.get("transfer-encoding", "").lower() == "chunked"
        content_length_str = response_headers.get("content-length")

        if is_stream:
            _send_frame(conn, {"event": "start", "request_id": request_id, "status": status})
            try:
                if is_chunked:
                    leftover = b""
                    for chunk in reader.iter_chunked():
                        # An SSE chunk may contain one or more events; split on \n\n.
                        # Buffer leftover bytes from the previous chunk to re-stitch.
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
                    # Some upstreams return non-chunked SSE — read until close, split.
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

            op = msg.get("op")
            if op == "attest":
                t0 = time.monotonic()
                user_data = _decode_b64(msg.get("user_data_b64"))
                nonce = _decode_b64(msg.get("nonce_b64"))
                # When the parent is going to use this doc as a KMS Decrypt
                # Recipient, it asks for the enclave's own RSA pubkey to be
                # baked in — KMS will return ciphertext-for-recipient that
                # only this running enclave's matching priv key can unwrap.
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
            if not isinstance(categories, list) or not categories:
                # No categories enabled → nothing to evaluate. Skip the
                # forward pass entirely so observation deployments don't
                # pay Granite latency for verdicts they wouldn't act on.
                conn.sendall(
                    (
                        json.dumps(
                            {
                                "request_id": msg.get("request_id"),
                                "verdict": "safe",
                                "label": None,
                                "per_category": {},
                                "latency_ms": 0.0,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                continue
            result = classifier.classify(
                text=msg.get("text", ""),
                direction=msg.get("direction", "input"),
                categories=[str(c) for c in categories],
                combined=bool(msg.get("combined", True)),
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            payload = {
                "request_id": msg.get("request_id"),
                "verdict": result["verdict"],
                "label": result["label"],
                "per_category": result["per_category"],
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


def main() -> int:
    import threading

    log.info("python=%s exe=%s", sys.version.split()[0], sys.executable)
    log.info("sys.path[:5]=%s", sys.path[:5])
    _load_models_from_parent()
    classifier = DualClassifier()

    AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
    VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)

    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, CLASSIFY_PORT))
    sock.listen(64)
    log.info("listening on vsock cid=ANY port=%d", CLASSIFY_PORT)

    while True:
        conn, addr = sock.accept()
        log.info("accepted from cid=%s port=%s", addr[0], addr[1])
        # Each parent-side client (NsmAttestationClient, VsockClassifier,
        # VsockChatClient) keeps a long-lived vsock connection. Without
        # threading, the first connection blocks the listener forever and
        # no other client can talk to us. Daemon thread per connection.
        threading.Thread(
            target=_serve_one, args=(conn, classifier), daemon=True
        ).start()


if __name__ == "__main__":
    sys.exit(main())

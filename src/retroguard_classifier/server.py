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

import base64
import hashlib
import json
import logging
import os
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
QWEN_PATH = MODEL_DIR / "qwen3guard.gguf"
GRANITE_SHA256 = os.environ.get("RG_GRANITE_SHA256", "").lower()
QWEN_SHA256 = os.environ.get("RG_QWEN_SHA256", "").lower()
MODEL_CTX = int(os.environ.get("RG_MODEL_CTX", "4096"))
MODEL_THREADS = int(os.environ.get("RG_MODEL_THREADS", "32"))


def _build_granite_prompt(text: str, direction: str) -> str:
    """Granite Guardian chat template (single-token verdict).

    IBM's recommended usage drives the model to a one-token "Yes"/"No"
    verdict per risk, not the multi-category enumeration our earlier
    prompt asked for. Trimming the system message (~100 → ~25 tokens of
    prompt overhead) cut warm latency from ~2.3 s to ~0.6 s in our
    benchmarks. Categories beyond a generic "harm" label are produced
    by Qwen3Guard on the output path.
    """
    speaker = "user" if direction == "input" else "assistant"
    return (
        "<|start_of_role|>system<|end_of_role|>"
        f"Reply 'Yes' if the {speaker} message is harmful, 'No' otherwise."
        "<|end_of_text|>\n"
        f"<|start_of_role|>{speaker}<|end_of_role|>{text}<|end_of_text|>\n"
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

    def __init__(
        self,
        name: str,
        model_path: Path,
        prompt_builder,
        stop_tokens: list[str],
        max_tokens: int,
        default_label: str,
    ) -> None:
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
        self._max_tokens = max_tokens
        self._default_label = default_label
        log.info("loaded %s", name)

    def classify(self, text: str, direction: str) -> dict[str, Any]:
        prompt = self._prompt_builder(text, direction)
        out = self._llm(
            prompt,
            max_tokens=self._max_tokens,
            temperature=0.0,
            stop=self._stop_tokens,
        )
        raw = (out.get("choices", [{}])[0].get("text") or "").strip()
        first_line = raw.splitlines()[0].strip().lower() if raw else ""
        unsafe = first_line.startswith("yes") or first_line.startswith("unsafe")
        verdict = "unsafe" if unsafe else "safe"
        label: str | None = None
        if verdict == "unsafe":
            if len(raw.splitlines()) > 1:
                label = raw.splitlines()[1].split(",")[0].strip().lower() or None
            label = label or self._default_label
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
            max_tokens=4,
            default_label="harm",
        )
        self.qwen: GuardrailEngine | None = None
        if QWEN_PATH.exists():
            self.qwen = GuardrailEngine(
                "qwen3guard-gen", QWEN_PATH, _build_qwen3guard_prompt,
                stop_tokens=["<|im_end|>", "</s>"],
                max_tokens=24,
                default_label="harm",
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
            _stream_blob(conn, QWEN_PATH, QWEN_SHA256, "qwen")
        elif mode == "kms":
            data_key = _negotiate_kms_data_key(conn, header)
            _stream_encrypted_blob(conn, GRANITE_PATH, data_key, GRANITE_SHA256, "granite")
            _stream_encrypted_blob(conn, QWEN_PATH, data_key, QWEN_SHA256, "qwen")
        else:
            raise RuntimeError(f"unknown load mode: {mode!r}")

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

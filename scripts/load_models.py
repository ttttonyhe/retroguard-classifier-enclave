#!/usr/bin/env python3
"""Stream the configured GGUF set into a freshly-booted enclave.

Two modes:

  Plaintext (default) — stream cleartext GGUFs; the enclave hashes each
  on the way in and refuses any that don't match the SHA-256 baked into
  the EIF. Suitable for public weights (Granite Guardian / Qwen3Guard).

  KMS-attested (`--mode kms`) — for sealed weights:
    1. We send `{"mode":"kms","models":[{"label":"granite"}, ...]}` and
       receive an attestation document (which embeds the enclave's
       fresh recipient pubkey).
    2. We hand the doc + the KMS-wrapped data key to KMS Decrypt with
       `Recipient.AttestationDocument`. KMS returns the data key
       re-encrypted to the enclave's pubkey.
    3. We forward the recipient ciphertext to the enclave; it unwraps
       to a 32-byte AES-256 key and uses it to decrypt each framed
       `[12B IV][ciphertext][16B tag]` model blob.

Wire format on RG_LOAD_PORT (default 5006):

    SEND: {"mode":"plaintext","models":[{"label":"granite"}, ...]}\\n
      then for each label (in order): [8-byte BE length][bytes]

    SEND: {"mode":"kms","models":[...],"nonce_b64":"..."}\\n
      RECV: {"attestation_doc_b64":"..."}\\n
      SEND: {"ciphertext_for_recipient_b64":"..."}\\n
      then for each label (in order): [8-byte BE framed-len][12B IV][ciphertext][16B tag]

In both modes the enclave finishes with `{"status":"loaded"}\\n` and
opens RG_VSOCK_PORT (default 5005) for classify traffic.

Usage (KMS):
    python3 load_models.py --mode kms \\
        --kms-key-id alias/retroguard-models \\
        --encrypted-data-key /opt/models/encrypted/data-key.kms \\
        --model granite=/opt/models/encrypted/granite.gcm \\
        --model qwen_06b=/opt/models/encrypted/qwen_06b.gcm \\
        --model qwen_4b=/opt/models/encrypted/qwen_4b.gcm \\
        --model qwen_8b=/opt/models/encrypted/qwen_8b.gcm

Usage (plaintext):
    python3 load_models.py --mode plaintext \\
        --model granite=/opt/models/granite.gguf \\
        --model qwen_06b=/opt/models/qwen3guard-gen-0.6b.gguf \\
        ...

Hash printing helper (for setting --build-arg <LABEL>_SHA256 on the EIF):
    python3 load_models.py --print-hashes \\
        --model granite=/opt/models/granite.gguf \\
        --model qwen_06b=/opt/models/qwen3guard-gen-0.6b.gguf \\
        ...
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
# Match the in-enclave receiver's chunk size; see server.py's RECV_CHUNK
# comment for the Linux 4.14 vsock-corruption rationale.
CHUNK = 1 << 14


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _send_blob(sock: socket.socket, path: Path, label: str) -> None:
    size = path.stat().st_size
    print(f"[{label}] streaming {size} bytes from {path}", flush=True)
    sock.sendall(struct.pack(">Q", size))
    sent = 0
    t0 = time.monotonic()
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            sock.sendall(chunk)
            h.update(chunk)
            sent += len(chunk)
    elapsed = time.monotonic() - t0
    mb_s = (sent / (1 << 20)) / max(elapsed, 1e-6)
    print(
        f"[{label}] streamed {sent} bytes in {elapsed:.1f}s ({mb_s:.1f} MiB/s) "
        f"sender_sha={h.hexdigest()}",
        flush=True,
    )


def _send_encrypted_blob(sock: socket.socket, path: Path, label: str) -> None:
    """Send a [12B IV][ciphertext][16B tag] blob produced by encrypt_models.py."""
    size = path.stat().st_size
    if size < 12 + 16:
        raise SystemExit(f"{label}: encrypted file too short ({size} bytes)")
    print(f"[{label}] streaming {size} encrypted bytes from {path}", flush=True)
    sock.sendall(struct.pack(">Q", size))
    sent = 0
    t0 = time.monotonic()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            sock.sendall(chunk)
            sent += len(chunk)
    elapsed = time.monotonic() - t0
    mb_s = (sent / (1 << 20)) / max(elapsed, 1e-6)
    print(f"[{label}] streamed {sent} bytes in {elapsed:.1f}s ({mb_s:.1f} MiB/s)", flush=True)


def _recv_line(sock: socket.socket) -> str:
    buf = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("enclave closed mid-handshake")
        buf.extend(chunk)
        if b"\n" in buf:
            return bytes(buf).split(b"\n", 1)[0].decode()


def _kms_handshake(
    sock: socket.socket,
    *,
    kms_key_id: str,
    encrypted_data_key: bytes,
    region: str,
    nonce: bytes | None,
    labels: list[str],
) -> None:
    """Send {"mode":"kms",...}, get attestation, call KMS, send back ciphertext_for_recipient."""
    import boto3  # type: ignore[import-not-found]

    header: dict[str, object] = {
        "mode": "kms",
        "models": [{"label": label} for label in labels],
    }
    if nonce:
        header["nonce_b64"] = base64.b64encode(nonce).decode()
    sock.sendall((json.dumps(header) + "\n").encode())

    line = _recv_line(sock)
    reply = json.loads(line)
    doc_b64 = reply.get("attestation_doc_b64")
    if not doc_b64:
        raise SystemExit(f"enclave returned no attestation_doc_b64: {reply!r}")
    attestation = base64.b64decode(doc_b64)
    print(f"[kms] got attestation doc: {len(attestation)} bytes", flush=True)

    kms = boto3.client("kms", region_name=region)
    t0 = time.monotonic()
    response = kms.decrypt(
        CiphertextBlob=encrypted_data_key,
        KeyId=kms_key_id,
        Recipient={
            "AttestationDocument": attestation,
            "KeyEncryptionAlgorithm": "RSAES_OAEP_SHA_256",
        },
    )
    elapsed = (time.monotonic() - t0) * 1000
    cfr = response["CiphertextForRecipient"]
    print(f"[kms] Decrypt OK in {elapsed:.0f}ms; CiphertextForRecipient={len(cfr)} bytes", flush=True)

    sock.sendall((json.dumps({"ciphertext_for_recipient_b64": base64.b64encode(cfr).decode()}) + "\n").encode())


def _parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model expects label=path, got {value!r}")
    label, _, path_str = value.partition("=")
    label = label.strip()
    path = Path(path_str.strip()).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"--model {value!r}: empty label")
    return label, path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cid", type=int, default=16)
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--mode", choices=["plaintext", "kms"], default="plaintext")
    p.add_argument(
        "--model", action="append", type=_parse_model_arg, metavar="LABEL=PATH",
        help="Repeatable: label=path of the GGUF (plaintext) or GCM blob (kms) to send.",
    )
    # KMS-mode only:
    p.add_argument("--kms-key-id", help="ARN/ID of the KMS CMK that wraps the data key")
    p.add_argument("--encrypted-data-key", type=Path, help="File containing the KMS-wrapped data key")
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--nonce-hex", help="Optional nonce to bind into the attestation doc")
    p.add_argument("--print-hashes", action="store_true",
                   help="Compute and print SHA-256 of each plaintext model "
                        "(for setting --build-arg <LABEL>_SHA256 on the EIF).")
    args = p.parse_args()

    if not args.model:
        print("at least one --model label=path is required", file=sys.stderr)
        return 2

    if args.print_hashes:
        for label, path in args.model:
            if not path.exists():
                print(f"missing: {path}", file=sys.stderr)
                return 2
            print(f"RG_{label.upper()}_SHA256={_sha256(path)}")
        return 0

    for _, path in args.model:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    if args.mode == "kms":
        for required in ("kms_key_id", "encrypted_data_key"):
            if not getattr(args, required):
                print(f"kms mode needs --{required.replace('_', '-')}", file=sys.stderr)
                return 2
        if not args.encrypted_data_key.exists():
            print(f"missing: {args.encrypted_data_key}", file=sys.stderr)
            return 2

    print(f"connecting to enclave cid={args.cid} port={args.port}", flush=True)
    s = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    s.connect((args.cid, args.port))
    try:
        labels = [label for label, _ in args.model]
        if args.mode == "plaintext":
            header = {
                "mode": "plaintext",
                "models": [{"label": label} for label in labels],
            }
            s.sendall((json.dumps(header) + "\n").encode())
            for label, path in args.model:
                _send_blob(s, path, label)
        else:
            nonce = bytes.fromhex(args.nonce_hex) if args.nonce_hex else None
            edk = args.encrypted_data_key.read_bytes()
            _kms_handshake(
                s,
                kms_key_id=args.kms_key_id,
                encrypted_data_key=edk,
                region=args.region,
                nonce=nonce,
                labels=labels,
            )
            for label, path in args.model:
                _send_encrypted_blob(s, path, label)

        ack = _recv_line(s)
        print(f"enclave ack: {ack!r}", flush=True)
        if '"status":"loaded"' not in ack:
            print("enclave did not confirm load — check enclave console", file=sys.stderr)
            return 3
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

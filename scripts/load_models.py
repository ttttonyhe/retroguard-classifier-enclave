#!/usr/bin/env python3
"""Stream Granite + Qwen3Guard GGUFs into a freshly-booted enclave.

Two modes:

  Plaintext (default) — stream cleartext GGUFs; the enclave hashes each
  on the way in and refuses any that don't match the SHA-256 baked into
  the EIF. Suitable for public weights (Granite Guardian / Qwen3Guard).

  KMS-attested (`--mode kms`) — for sealed weights:
    1. We send `{"mode":"kms"}` and receive an attestation document
       (which embeds the enclave's fresh recipient pubkey).
    2. We hand the doc + the KMS-wrapped data key to KMS Decrypt with
       `Recipient.AttestationDocument`. KMS returns the data key
       re-encrypted to the enclave's pubkey.
    3. We forward the recipient ciphertext to the enclave; it unwraps
       to a 32-byte AES-256 key and uses it to decrypt each framed
       `[12B IV][ciphertext][16B tag]` model blob.

Wire format on RG_LOAD_PORT (default 5006):

    SEND: {"mode":"plaintext"}\\n
      then for each model: [8-byte BE length][bytes]

    SEND: {"mode":"kms","nonce_b64":"..."}\\n
      RECV: {"attestation_doc_b64":"..."}\\n
      SEND: {"ciphertext_for_recipient_b64":"..."}\\n
      then for each model: [8-byte BE framed-len][12B IV][ciphertext][16B tag]

In both modes the enclave finishes with `{"status":"loaded"}\\n` and
opens RG_VSOCK_PORT (default 5005) for classify traffic.
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
    """Send a [12B IV][ciphertext][16B tag] blob produced by encrypt_models.py.

    The framed length on the wire is the file size (IV + ct + tag), so
    the enclave can pre-read IV/tag at known offsets.
    """
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
) -> None:
    """Send {"mode":"kms"}, get attestation, call KMS, send back ciphertext_for_recipient."""
    import boto3  # type: ignore[import-not-found]

    header = {"mode": "kms"}
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cid", type=int, default=16)
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--mode", choices=["plaintext", "kms"], default="plaintext")
    p.add_argument("--granite", type=Path, required=False)
    p.add_argument("--qwen", type=Path, required=False)
    # KMS-mode only:
    p.add_argument("--kms-key-id", help="ARN/ID of the KMS CMK that wraps the data key")
    p.add_argument("--encrypted-data-key", type=Path, help="File containing the KMS-wrapped data key")
    p.add_argument("--encrypted-granite", type=Path, help="AES-GCM ciphertext of granite GGUF (IV||ct||tag)")
    p.add_argument("--encrypted-qwen", type=Path, help="AES-GCM ciphertext of qwen GGUF (IV||ct||tag)")
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--nonce-hex", help="Optional nonce to bind into the attestation doc")
    p.add_argument("--print-hashes", action="store_true",
                   help="Compute and print SHA-256 of plaintext models, then exit "
                        "(for setting --build-arg GRANITE_SHA256/QWEN_SHA256 on the EIF).")
    args = p.parse_args()

    if args.print_hashes:
        if not args.granite or not args.qwen:
            print("--print-hashes needs --granite and --qwen", file=sys.stderr)
            return 2
        print(f"GRANITE_SHA256={_sha256(args.granite)}")
        print(f"QWEN_SHA256={_sha256(args.qwen)}")
        return 0

    if args.mode == "plaintext":
        if not args.granite or not args.qwen:
            print("plaintext mode needs --granite and --qwen", file=sys.stderr)
            return 2
        for path in (args.granite, args.qwen):
            if not path.exists():
                print(f"missing: {path}", file=sys.stderr)
                return 2
    else:
        for required in ("kms_key_id", "encrypted_data_key", "encrypted_granite", "encrypted_qwen"):
            if not getattr(args, required):
                print(f"kms mode needs --{required.replace('_', '-')}", file=sys.stderr)
                return 2
        for path in (args.encrypted_data_key, args.encrypted_granite, args.encrypted_qwen):
            if not path.exists():
                print(f"missing: {path}", file=sys.stderr)
                return 2

    print(f"connecting to enclave cid={args.cid} port={args.port}", flush=True)
    s = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    s.connect((args.cid, args.port))
    try:
        if args.mode == "plaintext":
            s.sendall(b'{"mode":"plaintext"}\n')
            _send_blob(s, args.granite, "granite")
            _send_blob(s, args.qwen, "qwen")
        else:
            nonce = bytes.fromhex(args.nonce_hex) if args.nonce_hex else None
            edk = args.encrypted_data_key.read_bytes()
            _kms_handshake(
                s,
                kms_key_id=args.kms_key_id,
                encrypted_data_key=edk,
                region=args.region,
                nonce=nonce,
            )
            _send_encrypted_blob(s, args.encrypted_granite, "granite")
            _send_encrypted_blob(s, args.encrypted_qwen, "qwen")

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

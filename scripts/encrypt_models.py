#!/usr/bin/env python3
"""One-shot helper to wrap GGUFs for the KMS-decrypt load path.

Run on a trusted box (parent host or build server) before deploying:

    python3 encrypt_models.py \\
        --kms-key-id alias/retroguard-models \\
        --granite /opt/models/granite-q4km.gguf \\
        --qwen    /opt/models/qwen3guard-q4km.gguf \\
        --out-dir /opt/models/encrypted

Produces three files in `--out-dir`:

    granite.gcm   = [12-byte IV][AES-GCM ciphertext][16-byte tag]
    qwen.gcm      = [12-byte IV][AES-GCM ciphertext][16-byte tag]
    data-key.kms  = KMS-wrapped 32-byte AES-256 data key (binary)

Then on the parent at boot:

    python3 load_models.py --mode kms \\
        --kms-key-id alias/retroguard-models \\
        --encrypted-data-key /opt/models/encrypted/data-key.kms \\
        --encrypted-granite  /opt/models/encrypted/granite.gcm \\
        --encrypted-qwen     /opt/models/encrypted/qwen.gcm
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

CHUNK = 1 << 16


def _aes_gcm_encrypt_to_file(src: Path, dst: Path, key: bytes) -> None:
    """Encrypt `src` with AES-256-GCM to `dst` as IV||ciphertext||tag.

    cryptography's AESGCM is one-shot — fine for ~5 GB models on a host
    with 32+ GiB RAM, which is the only place this runs.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]

    iv = secrets.token_bytes(12)  # 96-bit IV is the GCM default
    plaintext = src.read_bytes()
    aes = AESGCM(key)
    ct_with_tag = aes.encrypt(iv, plaintext, None)
    # cryptography appends the 16-byte tag at the end of ct_with_tag.
    with dst.open("wb") as f:
        f.write(iv)
        f.write(ct_with_tag)
    print(f"[{src.name}] {len(plaintext)} pt → {dst.stat().st_size} ct ({dst})", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kms-key-id", required=True, help="ARN/Alias/ID of the wrapping CMK")
    p.add_argument("--granite", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = p.parse_args()

    import boto3  # type: ignore[import-not-found]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[kms] generating 256-bit data key from {args.kms_key_id}", flush=True)
    kms = boto3.client("kms", region_name=args.region)
    response = kms.generate_data_key(KeyId=args.kms_key_id, KeySpec="AES_256")
    plaintext_dk: bytes = response["Plaintext"]
    encrypted_dk: bytes = response["CiphertextBlob"]
    print(f"[kms] data key: plaintext={len(plaintext_dk)}B  wrapped={len(encrypted_dk)}B", flush=True)

    edk_path = args.out_dir / "data-key.kms"
    edk_path.write_bytes(encrypted_dk)
    print(f"[kms] wrote wrapped data key → {edk_path}", flush=True)

    try:
        _aes_gcm_encrypt_to_file(args.granite, args.out_dir / "granite.gcm", plaintext_dk)
        _aes_gcm_encrypt_to_file(args.qwen, args.out_dir / "qwen.gcm", plaintext_dk)
    finally:
        # Best-effort wipe of the in-memory key (Python's GC will collect
        # the bytes object eventually; this just shortens the window).
        del plaintext_dk

    print("[done] ciphertext models + wrapped data key are in", args.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

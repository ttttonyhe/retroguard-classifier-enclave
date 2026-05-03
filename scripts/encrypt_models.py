#!/usr/bin/env python3
"""One-shot helper to wrap GGUFs for the KMS-decrypt load path.

Wraps N model files with a single AES-256 data key (shared across
files); the data key itself is wrapped by the KMS CMK so only the
attested enclave can recover it.

Usage:

    python3 encrypt_models.py \\
        --kms-key-id alias/retroguard-models \\
        --model granite=/opt/models/granite-q4ks.gguf \\
        --model qwen_06b=/opt/models/qwen3guard-gen-0.6b-q4km.gguf \\
        --model qwen_4b=/opt/models/qwen3guard-gen-4b-q4km.gguf \\
        --model qwen_8b=/opt/models/qwen3guard-gen-8b-q4km.gguf \\
        --out-dir /opt/models/encrypted

Produces in `--out-dir`:

    granite.gcm   = [12-byte IV][AES-GCM ciphertext][16-byte tag]
    qwen_06b.gcm  = ...
    qwen_4b.gcm   = ...
    qwen_8b.gcm   = ...
    data-key.kms  = KMS-wrapped 32-byte AES-256 data key (binary)

Then on the parent at boot:

    python3 load_models.py --mode kms \\
        --kms-key-id alias/retroguard-models \\
        --encrypted-data-key /opt/models/encrypted/data-key.kms \\
        --model granite=/opt/models/encrypted/granite.gcm \\
        --model qwen_06b=/opt/models/encrypted/qwen_06b.gcm \\
        --model qwen_4b=/opt/models/encrypted/qwen_4b.gcm \\
        --model qwen_8b=/opt/models/encrypted/qwen_8b.gcm

Per-IV reuse note: each file gets a fresh 12-byte IV (96-bit GCM
default). AES-GCM is safe up to ~2^32 IVs per key when IVs are
random; four IVs is trivially within bounds.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

CHUNK = 4 * 1024 * 1024  # 4 MiB — keeps multi-GB GGUFs out of single-shot AESGCM limits


def _aes_gcm_encrypt_to_file(src: Path, dst: Path, key: bytes) -> None:
    """Encrypt `src` with AES-256-GCM to `dst` as IV||ciphertext||tag."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]

    iv = secrets.token_bytes(12)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()

    src_size = src.stat().st_size
    written = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(iv)
        while True:
            chunk = fin.read(CHUNK)
            if not chunk:
                break
            fout.write(encryptor.update(chunk))
            written += len(chunk)
        fout.write(encryptor.finalize())
        fout.write(encryptor.tag)
    print(f"[{src.name}] {src_size} pt → {dst.stat().st_size} ct ({dst})", flush=True)


def _parse_model_arg(value: str) -> tuple[str, Path]:
    """Parse a --model label=path arg into (label, Path)."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model expects label=path, got {value!r}")
    label, _, path_str = value.partition("=")
    label = label.strip()
    path = Path(path_str.strip()).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"--model {value!r}: empty label")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"--model {value!r}: missing file {path}")
    return label, path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kms-key-id", required=True, help="ARN/Alias/ID of the wrapping CMK")
    p.add_argument(
        "--model", action="append", required=True, type=_parse_model_arg,
        metavar="LABEL=PATH",
        help="Repeatable: label=path for each plaintext GGUF to encrypt.",
    )
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
        for label, src_path in args.model:
            dst_path = args.out_dir / f"{label}.gcm"
            _aes_gcm_encrypt_to_file(src_path, dst_path, plaintext_dk)
    finally:
        del plaintext_dk

    print(f"[done] {len(args.model)} ciphertext model(s) + wrapped data key in {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

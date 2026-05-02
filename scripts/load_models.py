#!/usr/bin/env python3
"""Stream Granite + Qwen3Guard GGUFs into a freshly-booted enclave.

Wire format on RG_LOAD_PORT (default 5006) — back-to-back framed blobs:

    [8-byte BE length][bytes]   Granite GGUF
    [8-byte BE length][bytes]   Qwen3Guard GGUF

The enclave verifies each blob's SHA-256 against constants baked
into the EIF (and therefore measured by PCR0). On verification
success the enclave replies with `{"status":"loaded"}\\n` and opens
RG_VSOCK_PORT (default 5005) for classify traffic.

Usage:
    python3 load_models.py \\
        --cid 16 \\
        --granite /opt/models/granite-guardian-4.1-8b.Q4_K_M.gguf \\
        --qwen    /opt/models/Qwen3Guard-Gen-8B.Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import hashlib
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
    print(f"[{label}] streamed {sent} bytes in {elapsed:.1f}s ({mb_s:.1f} MiB/s) sender_sha={h.hexdigest()}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cid", type=int, default=16)
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--granite", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--print-hashes", action="store_true",
                   help="Compute and print SHA-256 of each model, then exit "
                        "(use the values to populate --build-arg "
                        "GRANITE_SHA256/QWEN_SHA256 when building the EIF).")
    args = p.parse_args()

    for path in (args.granite, args.qwen):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    if args.print_hashes:
        print(f"GRANITE_SHA256={_sha256(args.granite)}")
        print(f"QWEN_SHA256={_sha256(args.qwen)}")
        return 0

    print(f"connecting to enclave cid={args.cid} port={args.port}", flush=True)
    s = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    s.connect((args.cid, args.port))
    try:
        _send_blob(s, args.granite, "granite")
        _send_blob(s, args.qwen, "qwen")
        ack = s.recv(256)
        print(f"enclave ack: {ack!r}", flush=True)
        if b'"status":"loaded"' not in ack:
            print("enclave did not confirm load — check enclave console", file=sys.stderr)
            return 3
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

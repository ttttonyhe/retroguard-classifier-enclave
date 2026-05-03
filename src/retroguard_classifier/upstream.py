"""TLS-over-vsock HTTP/1.1 client for in-enclave upstream calls.

The Nitro Enclave has only vsock; no IP networking, no DNS. The parent
EC2 host runs `vsock-proxy` daemons that bridge a vsock port to one
allowlisted upstream (api.openai.com:443, api.anthropic.com:443).

This module:

  * Opens a vsock socket to (parent_cid=3, port=N)
  * Wraps it with `ssl.create_default_context().wrap_socket(...)`,
    passing `server_hostname=upstream_host` so SNI works and the cert
    chain is verified against the bundled CA store. The TLS handshake
    happens *inside* the enclave — vsock-proxy never sees plaintext.
  * Sends an HTTP/1.1 request and parses the response head.
  * Exposes `iter_chunks()` for SSE streams (transfer-encoding: chunked
    with one SSE event per chunk, which matches OpenAI and Anthropic).
  * Exposes `read_body()` for buffered responses.

We hand-roll HTTP/1.1 because httpx/urllib3 want to manage their own
sockets — neither has a clean API for "use this pre-connected socket".
The protocol surface we need is small: POST with JSON body, parse a
status + headers + chunked body. ~150 lines vs. a dependency that
would need a custom transport adapter on top of pre-built sockets.
"""

from __future__ import annotations

import logging
import socket
import ssl
from collections.abc import Iterator
from typing import Optional

log = logging.getLogger("enclave.upstream")

AF_VSOCK = getattr(socket, "AF_VSOCK", 40)

# Nitro Enclaves: parent host is always CID=3 from the enclave's view.
PARENT_CID = 3

# Upstream → parent vsock port. The parent's vsock-proxy systemd units
# bind these and forward to the matching public hostname:443.
DEFAULT_UPSTREAM_PORTS: dict[str, int] = {
    "api.openai.com": 8001,
    "api.anthropic.com": 8002,
}

_HTTP_CHUNK_SIZE = 16 * 1024  # 16 KiB; vsock recv() pre-Linux-5.6 capped at 16 KiB anyway


class HttpError(Exception):
    """Raised on protocol-level failures talking to upstream."""


def open_tls_over_vsock(
    *,
    upstream_host: str,
    vsock_port: int,
    parent_cid: int = PARENT_CID,
    timeout: float = 60.0,
) -> ssl.SSLSocket:
    """Open a TLS-wrapped socket to the upstream via the parent's vsock-proxy."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((parent_cid, vsock_port))
    ctx = ssl.create_default_context()
    # Strict cert verification — this is the whole point of moving TLS into
    # the enclave. If it fails, fail closed.
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx.wrap_socket(sock, server_hostname=upstream_host)


def send_request(
    tls: ssl.SSLSocket,
    *,
    method: str,
    path: str,
    host: str,
    headers: dict[str, str],
    body: bytes,
) -> None:
    """Frame an HTTP/1.1 request and write it to the TLS socket."""
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
    for k, v in headers.items():
        lines.append(f"{k}: {v}")
    if body:
        lines.append(f"Content-Length: {len(body)}")
    # Always close after one request — keeps the chunked-body parser simple
    # and avoids a connection pool inside the enclave.
    lines.append("Connection: close")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    tls.sendall(head)
    if body:
        tls.sendall(body)


class HttpReader:
    """Buffered reader on top of a TLS socket, just enough HTTP/1.1 to
    consume an OpenAI/Anthropic response.

    Supports:
      * status line + header parsing
      * `Transfer-Encoding: chunked` body iteration (each chunk yielded
        verbatim — fine for SSE since each frame is its own chunk)
      * `Content-Length` body read
      * Connection-close framing (read until EOF)
    """

    def __init__(self, sock: ssl.SSLSocket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def _recv(self) -> bytes:
        return self._sock.recv(_HTTP_CHUNK_SIZE)

    def _read_line(self) -> Optional[bytes]:
        while True:
            i = self._buf.find(b"\r\n")
            if i >= 0:
                line = bytes(self._buf[:i])
                del self._buf[: i + 2]
                return line
            chunk = self._recv()
            if not chunk:
                if self._buf:
                    line = bytes(self._buf)
                    self._buf.clear()
                    return line
                return None
            self._buf.extend(chunk)

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._recv()
            if not chunk:
                break
            self._buf.extend(chunk)
        if len(self._buf) < n:
            raise HttpError(f"upstream closed mid-body, wanted {n}, have {len(self._buf)}")
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def read_status(self) -> int:
        line = self._read_line()
        if not line:
            raise HttpError("upstream closed before sending status line")
        parts = line.decode("iso-8859-1", errors="replace").split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise HttpError(f"bad status line: {line!r}")
        return int(parts[1])

    def read_headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        while True:
            line = self._read_line()
            if line is None:
                raise HttpError("upstream closed inside header block")
            if line == b"":
                return out
            k, _, v = line.decode("iso-8859-1", errors="replace").partition(":")
            out[k.strip().lower()] = v.strip()

    def iter_chunked(self) -> Iterator[bytes]:
        """Yield one body chunk per Transfer-Encoding chunk."""
        while True:
            size_line = self._read_line()
            if size_line is None:
                return
            try:
                size = int(size_line.split(b";", 1)[0], 16)
            except ValueError as e:
                raise HttpError(f"bad chunk size {size_line!r}") from e
            if size == 0:
                # Trailers (we ignore) + final empty line
                while True:
                    trail = self._read_line()
                    if trail is None or trail == b"":
                        return
            data = self._read_exact(size)
            self._read_exact(2)  # trailing CRLF
            yield data

    def read_until_close(self) -> bytes:
        out = bytearray(self._buf)
        self._buf.clear()
        while True:
            chunk = self._recv()
            if not chunk:
                return bytes(out)
            out.extend(chunk)

    def read_fixed(self, n: int) -> bytes:
        return self._read_exact(n)

"""Unit tests for the hand-rolled HTTP/1.1 reader.

The enclave doesn't link httpx (TLS terminates over a vsock socket and
neither httpx nor urllib3 has a clean API for pre-built sockets), so
we hand-rolled a tiny reader. These tests exercise the corners that
matter for OpenAI/Anthropic responses:

  * Status + header parse
  * Transfer-Encoding: chunked iteration
  * Content-Length body read
  * Connection: close (read-until-EOF) body read
"""

from __future__ import annotations

import pytest

from retroguard_classifier.upstream import HttpError, HttpReader


class _MemorySocket:
    """Mock socket that doles out a fixed payload via recv()."""

    def __init__(self, payload: bytes, chunk_size: int = 64) -> None:
        self._buf = payload
        self._chunk = chunk_size

    def recv(self, n: int) -> bytes:
        if not self._buf:
            return b""
        take = min(n, self._chunk, len(self._buf))
        out, self._buf = self._buf[:take], self._buf[take:]
        return out


def test_status_and_headers() -> None:
    payload = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 13\r\n"
        b"\r\n"
        b'{"hello":"a"}'
    )
    reader = HttpReader(_MemorySocket(payload))  # type: ignore[arg-type]
    assert reader.read_status() == 200
    headers = reader.read_headers()
    assert headers["content-type"] == "application/json"
    assert headers["content-length"] == "13"
    assert reader.read_fixed(13) == b'{"hello":"a"}'


def test_chunked_iteration() -> None:
    payload = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n"
        b"6\r\n world\r\n"
        b"0\r\n\r\n"
    )
    reader = HttpReader(_MemorySocket(payload))  # type: ignore[arg-type]
    assert reader.read_status() == 200
    reader.read_headers()
    chunks = list(reader.iter_chunked())
    assert chunks == [b"hello", b" world"]


def test_chunked_handles_chunk_extension() -> None:
    # OpenAI sometimes sends `5;ext=value\r\n` — we ignore extensions.
    payload = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;ext=ok\r\nhello\r\n0\r\n\r\n"
    reader = HttpReader(_MemorySocket(payload))  # type: ignore[arg-type]
    reader.read_status()
    reader.read_headers()
    assert list(reader.iter_chunked()) == [b"hello"]


def test_read_until_close() -> None:
    payload = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nhello world"
    reader = HttpReader(_MemorySocket(payload))  # type: ignore[arg-type]
    reader.read_status()
    reader.read_headers()
    assert reader.read_until_close() == b"hello world"


def test_truncated_body_raises() -> None:
    # Content-Length says 100 but only 5 bytes follow.
    payload = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nhello"
    reader = HttpReader(_MemorySocket(payload))  # type: ignore[arg-type]
    reader.read_status()
    reader.read_headers()
    with pytest.raises(HttpError):
        reader.read_fixed(100)

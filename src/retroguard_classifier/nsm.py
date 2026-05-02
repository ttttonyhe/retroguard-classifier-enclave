"""Minimal /dev/nsm binding for in-enclave attestation.

The Nitro Secure Module exposes a single ioctl that takes a CBOR-encoded
request and returns a CBOR-encoded response. For attestation we send

    {"Attestation": {"user_data": ..., "nonce": ..., "public_key": ...}}

and get back

    {"Attestation": {"document": <COSE_Sign1 bytes>}}

The COSE_Sign1 document is what customers verify with @retroguard/verify
against the AWS Nitro root certificate. When `public_key` is supplied,
KMS's `Decrypt(Recipient=...)` operation will re-encrypt the plaintext
to that key — that's how the parent can hand the enclave a KMS-released
data key without ever seeing it itself.

Why not import aws_nitro_enclaves_nsm_api? It ships as a Rust crate;
the PyPI wheel is x86_64-only and doesn't pin a stable ABI. A 60-line
ctypes shim is easier to audit and adds no build-time complexity to
the EIF.
"""

from __future__ import annotations

import ctypes
import fcntl
import threading
from typing import Optional

import cbor2  # type: ignore[import-not-found]
from cryptography.hazmat.primitives import serialization  # type: ignore[import-not-found]
from cryptography.hazmat.primitives.asymmetric import rsa  # type: ignore[import-not-found]
from cryptography.hazmat.primitives.asymmetric.padding import MGF1, OAEP  # type: ignore[import-not-found]
from cryptography.hazmat.primitives.hashes import SHA256  # type: ignore[import-not-found]

# struct nsm_msg { struct iovec request; struct iovec response; }
# iovec on x86_64: { void* base; size_t length; }   → 16 bytes each → 32 total
class _IOVec(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
    ]


class _NSMMsg(ctypes.Structure):
    _fields_ = [
        ("request", _IOVec),
        ("response", _IOVec),
    ]


# _IOWR(0x0A, 0, struct nsm_msg) — see aws-nitro-enclaves-nsm-api/src/driver/include/nsm.h
# Linux ioctl encoding: (dir=3 << 30) | (size=32 << 16) | (type=0x0A << 8) | nr=0
_NSM_IOCTL_MSG = 0xC020_0A00

_RESPONSE_BUFFER_SIZE = 16 * 1024  # attestation docs are ~5 KB; 16 KB is generous


def get_attestation_document(
    *,
    user_data: Optional[bytes] = None,
    nonce: Optional[bytes] = None,
    public_key: Optional[bytes] = None,
) -> bytes:
    """Ask the NSM for a fresh COSE_Sign1 attestation document.

    `user_data` and `nonce` end up bound into the signed document so the
    customer can verify the doc was generated for their request (replay-
    resistant). `public_key` is meaningful when the customer wants to
    encrypt a payload to the enclave; we don't use it here.
    """
    request = {
        "Attestation": {
            "user_data": user_data,
            "nonce": nonce,
            "public_key": public_key,
        }
    }
    request_cbor = cbor2.dumps(request)

    # Pin the request bytes into memory the kernel can see.
    request_buf = (ctypes.c_ubyte * len(request_cbor)).from_buffer_copy(request_cbor)
    response_buf = (ctypes.c_ubyte * _RESPONSE_BUFFER_SIZE)()

    msg = _NSMMsg(
        request=_IOVec(
            base=ctypes.cast(request_buf, ctypes.c_void_p).value,
            length=len(request_cbor),
        ),
        response=_IOVec(
            base=ctypes.cast(response_buf, ctypes.c_void_p).value,
            length=_RESPONSE_BUFFER_SIZE,
        ),
    )

    with open("/dev/nsm", "rb+", buffering=0) as fd:
        fcntl.ioctl(fd.fileno(), _NSM_IOCTL_MSG, msg)

    response_bytes = bytes(response_buf[: msg.response.length])
    response = cbor2.loads(response_bytes)
    if "Attestation" not in response or "document" not in response["Attestation"]:
        raise RuntimeError(f"unexpected NSM response: {response!r}")
    return bytes(response["Attestation"]["document"])


# ---------------------------------------------------------------------------
# Per-enclave RSA keypair for KMS Decrypt(Recipient=...)
# ---------------------------------------------------------------------------
#
# We generate an RSA-2048 keypair at first use and keep the private key in
# enclave memory only. The public key is embedded in attestation documents
# so KMS can return data-key plaintext re-encrypted to a key that only this
# specific running enclave can unwrap. Rotated implicitly on every reboot.

_keypair_lock = threading.Lock()
_private_key: Optional[rsa.RSAPrivateKey] = None
_public_key_der: Optional[bytes] = None


def _ensure_keypair() -> tuple[rsa.RSAPrivateKey, bytes]:
    global _private_key, _public_key_der
    with _keypair_lock:
        if _private_key is None:
            _private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            _public_key_der = _private_key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        assert _public_key_der is not None  # for the type checker
        return _private_key, _public_key_der


def get_recipient_public_key_der() -> bytes:
    """Return the DER-encoded SubjectPublicKeyInfo for KMS Recipient use."""
    _, der = _ensure_keypair()
    return der


def unwrap_kms_recipient_ciphertext(ciphertext_for_recipient: bytes) -> bytes:
    """Decrypt KMS's `CiphertextForRecipient` using the matching private key.

    KMS encrypts with RSAES_OAEP_SHA_256 + MGF1-SHA256 when
    `KeyEncryptionAlgorithm: 'RSAES_OAEP_SHA_256'` is set on the
    Decrypt request — match that exactly.
    """
    priv, _ = _ensure_keypair()
    return priv.decrypt(
        ciphertext_for_recipient,
        OAEP(mgf=MGF1(algorithm=SHA256()), algorithm=SHA256(), label=None),
    )

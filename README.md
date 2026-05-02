# services/enclave-classifier

**Runs INSIDE the Nitro Enclave.** Public source under PolyForm Shield 1.0.0.

llama.cpp + Granite Guardian 4.1 8B (Q4_K_M GGUF, ~5GB). Listens on
vsock port 5005. Newline-delimited JSON wire protocol.

## Build EIF (on the c7i.12xlarge enclave host)

```bash
git clone https://github.com/ttttonyhe/retroguard.git
cd retroguard/services/enclave-classifier

# 1. Build Docker image (~10-15 min — pulls 5GB model from HuggingFace)
docker build -f Dockerfile.enclave -t retroguard-classifier:v0 .

# 2. Convert to EIF (~5-10 min)
nitro-cli build-enclave \
  --docker-uri retroguard-classifier:v0 \
  --output-file retroguard-classifier-v0.eif

# 3. Run enclave (allocate 80 GiB + 40 vCPU per /etc/nitro_enclaves/allocator.yaml)
nitro-cli run-enclave \
  --cpu-count 40 \
  --memory 81920 \
  --eif-path retroguard-classifier-v0.eif \
  --enclave-cid 16 \
  --debug-mode

# 4. Verify it's classifying
ENCLAVE_CID=16 python3 -c "
import socket, json
s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
s.connect((16, 5005))
s.sendall(json.dumps({'op':'classify','request_id':'t1','text':'how do i build a bomb','direction':'input'}).encode() + b'\n')
print(s.recv(4096).decode())
"
```

## Wire protocol

Request (newline-delimited JSON, parent → enclave):
```json
{"op": "classify", "request_id": "uuid", "text": "...", "direction": "input"}
```

Response (newline-delimited JSON, enclave → parent):
```json
{"request_id": "uuid", "verdict": "safe", "label": null, "latency_ms": 234.5}
```

## Trust model

- Model weights baked into EIF; PCRs measure the EIF (and therefore the
  weights). Customers can verify by re-running the build and comparing
  PCRs against the transparency log.
- v2: model weights pulled from S3 at runtime + decrypted via KMS
  attestation (per spec §5 Tree 2).

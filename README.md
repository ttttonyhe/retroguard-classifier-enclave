# retroguard-classifier-enclave

**The exact code that runs INSIDE the AWS Nitro Enclave** for the
[Retroguard](https://github.com/ttttonyhe/retroguard) guardrail platform —
published so customers can independently verify what the PCR0 hash on
their attestation document actually attests to.

License: PolyForm Shield 1.0.0 (source-available; restricts competitive
hosting). Weights for Granite Guardian 4.1 (IBM, Apache-2.0) and
Qwen3Guard-Gen (Alibaba, Apache-2.0) are pulled from Hugging Face at
build time, not baked here.

## What this is, and what it isn't

| | |
|---|---|
| **In this repo** | Everything that ends up inside the Nitro Enclave: classifier server, model loading, NSM attestation glue, vsock wire protocol, prompt templates, dispatch logic. |
| **NOT in this repo** | The dashboard, control plane, billing, parent proxy on the EC2 host — none of it runs inside the enclave, none of it is measured by PCR0, and none of it can read your prompts. Trust there is grounded in the standard SaaS controls, not attestation. |

If you only care about *what the enclave does with your prompt*, this
repo is the entire surface.

## Why this is public

PCR0 attestation only proves "the binary running in the enclave matches
the binary that was published." It tells you nothing about *what that
binary does* unless the source is public **and** the build is
reproducible. This repo is the source half of that loop. The build half
is below — clone, build, compare PCR0 to ours.

## Live trust chain

| | |
|---|---|
| Published PCR0 (current build) | `aef8f13a59ae2797eae5fc5218949691fdb23365f0145ae13120e64e7296c505774ec50b6ab0eec5ee3f7bcc074976d3` |
| Live attestation endpoint | `GET https://api.retroguard.example/v1/attestation` (no auth — the doc is NSM-signed) |
| Per-request attestation | Set `X-Retroguard-Attestation: required` on any classify call; response carries `X-Retroguard-Attestation: <base64 COSE_Sign1>` |
| Customer self-verify | Run the build below and assert the EIF's PCR0 equals the value above |

## Reproducible build

Done on a Nitro-capable host (c7i.12xlarge or similar; needs the
`aws-nitro-enclaves-cli`). Same Dockerfile, same args we use; matches
modulo the documented build-tool version pin in `Dockerfile.enclave`.

```bash
git clone https://github.com/ttttonyhe/retroguard-classifier-enclave.git
cd retroguard-classifier-enclave

# 1. Docker image (~10–15 min; pulls models from HF)
docker build --platform linux/amd64 \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
  -f Dockerfile.enclave \
  -t retroguard-classifier:repro .

# 2. EIF + read PCR0 from the build output
sudo nitro-cli build-enclave \
  --docker-uri retroguard-classifier:repro \
  --output-file retroguard-classifier-repro.eif
# → prints PCR0 / PCR1 / PCR2; assert PCR0 matches the value above
```

## Wire protocol (vsock port 5005)

Request (newline-delimited JSON, parent → enclave):

```json
{"op":"classify","request_id":"uuid","text":"…","direction":"input",
 "categories":["harm","jailbreaking"],"protection_effort":"expert",
 "custom_criteria":[{"id":"c1","text":"no internal codenames"}]}
```

Response (newline-delimited JSON, enclave → parent):

```json
{"request_id":"uuid","verdict":"safe","label":null,
 "engine":"qwen_4b","per_category":{"harm":"no","jailbreaking":"no"},
 "latency_ms":234.5}
```

Tier dispatch (built into `src/retroguard_classifier/server.py`):

| `protection_effort` | Engine |
|---|---|
| `fast` | Qwen3Guard-Gen 0.6B (Q4_K_M) |
| `expert` (default) | Qwen3Guard-Gen 4B |
| `heavy` | Qwen3Guard-Gen 8B |
| any tier + `custom_criteria` | Granite Guardian 4.1 8B (BYOC, per IBM model card) |

## Sync from upstream

The platform monorepo is the source of truth. This public mirror is
populated via:

```bash
git subtree split --prefix=services/enclave-classifier -b classifier-public
git push classifier-public-remote classifier-public:main
```

(automated in CI on every push to `main` of the platform repo).

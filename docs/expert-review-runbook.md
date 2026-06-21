# Expert Review Runbook

This runbook prepares the service for a controlled expert review session using Hybrid retrieval, Gemini generation, and the browser UI.

## Target Runtime

Use this configuration for the strongest measured retrieval path:

```bash
RETRIEVER=hybrid
GENERATOR=gemini
GEMINI_MODEL=gemini-2.5-flash
CHUNKS_PATH=.data/chunks/telco_v1.jsonl
TOP_K=5
MIN_SCORE=auto
HYBRID_SOURCE_K=100
HYBRID_RRF_C=40
HYBRID_VERTEX_WEIGHT=2.0
```

Hybrid retrieval requires a deployed Vertex AI Vector Search endpoint. Gemini generation requires Vertex AI credentials and `GCP_PROJECT_ID`, `REGION`, and `GEMINI_MODEL`.

## Pre-Session Checks

Run the readiness checker before sharing the UI:

```bash
set -a
. ./.env
set +a

python scripts/demo_readiness.py --strict
```

The checker verifies:

- generated chunks exist and are readable;
- Hybrid + Gemini runtime variables are set;
- Vertex AI Vector Search IDs are present;
- optional cloud dependencies are installed;
- the access model is not public unauthenticated access.

It does not create, deploy, or delete cloud resources.

## Local Review Session

For a local controlled session:

```bash
set -a
. ./.env
set +a

export RETRIEVER=hybrid
export GENERATOR=gemini
export GEMINI_MODEL=gemini-2.5-flash

python -m serving.app --chunks .data/chunks/telco_v1.jsonl --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080/
```

Use a short preflight set before the review:

- What are the three RLC modes?
- How does AM RLC do error recovery by retransmission after reception failure?
- Which MAC procedure tells the serving gNB how much uplink data is waiting in the UE?
- Which RRC clause covers detection of physical layer problems in RRC_CONNECTED?
- Which PDCP entity performs ciphering?

The last question should refuse or report insufficient evidence because PDCP is outside the indexed MAC/RLC/RRC corpus.

## Hosted Review Session

For hosted access, keep the same serving image and expose the browser UI only behind an identity-aware access layer.

Runtime requirements:

- provide the generated chunk JSONL to the container outside the public repository;
- set `CHUNKS_PATH` to the mounted or staged chunk file;
- set the Hybrid + Gemini environment variables from the target runtime section;
- keep Vertex AI Vector Search deployed for the review window;
- keep budget alerts and logs enabled;
- delete or undeploy the Vector Search endpoint after the review window if it is no longer needed.

Do not expose `/` or `/ask` as a public unauthenticated endpoint for expert review.

## Review Notes

Track issues using this shape:

```text
Question:
Expected behavior:
Observed answer:
Citation issue, if any:
Missing or incorrect technical detail:
Spec section that should support the answer, if known:
```

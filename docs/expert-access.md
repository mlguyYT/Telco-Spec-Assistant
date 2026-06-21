# Expert Access

This project should be tested by domain experts through a controlled, private endpoint before any broader exposure.

## Recommended Pilot Setup

Deploy the serving API to Cloud Run as a private service and grant the reviewer permission to invoke it with their Google account. This keeps the endpoint authenticated, avoids distributing local files, and lets the backend keep using the same `/ask` contract as local development.

The reviewer needs:

- the Cloud Run service URL;
- permission to invoke the service;
- a short scope note: currently MAC, RLC, and RRC Rel-19 v19.2.0;
- sample questions that cover exact lookup, paraphrase lookup, and out-of-scope refusal;
- a simple feedback template for wrong citation, incomplete answer, unsupported answer, and missing topic.

## Access Options

| Option | Use when | Tradeoff |
|---|---|---|
| Cloud Run IAM | Reviewer can use an authenticated HTTP client | Fastest secure pilot for the API |
| Cloud Run plus a small web UI | Reviewer should not use curl or Postman | Better usability, more implementation work |
| Identity-Aware Proxy | A browser-facing internal UI is needed | Strong access control, extra setup |
| Temporary tunnel | Short trusted live demo only | Easy to start, not suitable as durable access |

Do not expose `/ask` publicly without authentication during the pilot. The service can trigger model calls and should have budget, quota, and logging controls in place before external use.

## Operational Guardrails

- Use a least-privilege runtime service account.
- Keep `.env`, service account keys, and generated corpora out of the repository.
- Set a billing budget alert before enabling managed retrieval or generation.
- Keep Cloud Run logs enabled for debugging, but do not ask reviewers to submit confidential network data.
- Rate-limit or quota the endpoint before sharing it beyond a small pilot.
- Keep Vertex AI Vector Search deployed only for planned evaluation or review windows, then delete the endpoint when it is no longer needed.

## Minimal Feedback Template

```text
Question:
Expected behavior:
Observed answer:
Citation issue, if any:
Missing or incorrect technical detail:
Spec section that should support the answer, if known:
```

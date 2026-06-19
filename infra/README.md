# Infrastructure

Terraform will provision the managed cloud resources in Phase 2:

- enabled APIs
- least-privilege service account
- RAG Engine / Vector Search resources
- Cloud Run service

Target region: `us-central1`.

Phase 1 keeps infrastructure executable locally through the root `Dockerfile`. The container runs the `/health` and `/ask` API against a mounted chunk JSONL file and does not include downloaded specifications or generated chunks in the image.

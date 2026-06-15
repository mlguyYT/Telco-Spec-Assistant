# Ingestion

V1 ingestion parses the RLC seed document, creates clause-aware chunks, attaches citation metadata, and imports those chunks into the Google Cloud retrieval backend.

No downloaded source specifications belong in this directory.

Local chunk generation:

```bash
python -m ingestion.run --manifest specs/manifest.example.yaml --seed-dir ../input/3gpp-documents --no-download
```

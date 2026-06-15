# Scripts

Utility scripts for fetching public specifications and preparing local development data.

V1 target:

- `fetch_specs.py --manifest specs/manifest.example.yaml`
- `fetch_specs.py --manifest specs/manifest.example.yaml --seed-dir ../input/3gpp-documents --no-download`

Downloaded specifications must be written to an ignored local data directory and never committed.

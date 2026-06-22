#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env. Create it from .env.example and fill in local runtime values." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

export RETRIEVER="${RETRIEVER:-hybrid}"
export GENERATOR="${GENERATOR:-gemini}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
export CHUNKS_PATH="${CHUNKS_PATH:-.data/chunks/telco_v1.jsonl}"
export TOP_K="${TOP_K:-5}"
export MIN_SCORE="${MIN_SCORE:-auto}"
export HYBRID_SOURCE_K="${HYBRID_SOURCE_K:-100}"
export HYBRID_RRF_C="${HYBRID_RRF_C:-40}"
export HYBRID_VERTEX_WEIGHT="${HYBRID_VERTEX_WEIGHT:-2.0}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"

echo "Checking local demo readiness..."
.venv/bin/python scripts/demo_readiness.py --strict

echo "Checking deployed Vector Search endpoint..."
endpoint_json="$(mktemp)"
endpoint_error="$(mktemp)"
trap 'rm -f "$endpoint_json" "$endpoint_error"' EXIT
if ! gcloud ai index-endpoints describe "$VS_ENDPOINT_ID" \
  --region="${REGION:-us-central1}" \
  --project="$GCP_PROJECT_ID" \
  --format=json > "$endpoint_json" 2> "$endpoint_error"; then
  echo "Vector Search endpoint is not available: $VS_ENDPOINT_ID" >&2
  echo "Deploy or recreate it first, then update .env with the printed VS_* IDs:" >&2
  echo "  python scripts/create_vector_index.py" >&2
  exit 1
fi

.venv/bin/python - "$endpoint_json" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = os.environ.get("VS_DEPLOYED_INDEX_ID", "")
deployed = payload.get("deployedIndexes", [])
ids = {item.get("id") for item in deployed}
if expected not in ids:
    raise SystemExit(f"Endpoint exists, but deployed index {expected!r} is not active. Active IDs: {sorted(ids)}")
PY

lan_ip="$(
  ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}'
)"
if [[ -z "$lan_ip" ]]; then
  lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

echo
echo "Expert review UI:"
if [[ -n "$lan_ip" ]]; then
  echo "  http://${lan_ip}:${PORT}/"
fi
echo "  http://127.0.0.1:${PORT}/"
echo
echo "Preflight questions:"
echo "  - What are the three RLC modes?"
echo "  - How does AM RLC do error recovery by retransmission after reception failure?"
echo "  - Which MAC procedure tells the serving gNB how much uplink data is waiting in the UE?"
echo "  - Which RRC clause covers detection of physical layer problems in RRC_CONNECTED?"
echo "  - Which PDCP entity performs ciphering?"
echo
echo "Press Ctrl+C to stop the local UI server."
exec .venv/bin/python -m serving.app --chunks "$CHUNKS_PATH" --host "$HOST" --port "$PORT"

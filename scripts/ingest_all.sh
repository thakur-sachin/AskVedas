#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-change-me}"

curl -sS -X POST "${API_URL}/api/v1/admin/ingest" \
  -H 'Content-Type: application/json' \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" \
  -d '{"docs":[],"force_rebuild":false}'

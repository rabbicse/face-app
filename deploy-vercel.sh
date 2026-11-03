#!/usr/bin/env bash
# Deploy the API and the web app to Vercel (free Hobby plan) with a Qdrant Cloud database.
# See docs/deploy-free.md for what each step does. Safe to re-run: it updates the same two projects.
#
#   ./deploy-vercel.sh                    deploy both
#   ./deploy-vercel.sh api | web          deploy one part
#   API_PROJECT=my-api WEB_PROJECT=my-web ./deploy-vercel.sh
#
# Secrets are read from the environment, or from these git-ignored files:
#   .run/vercel-token.txt   VERCEL_TOKEN=...
#   .run/qdrant.env         QDRANT_URL=...  and  QDRANT_API_KEY=...
# bash 3.2 compatible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
WHAT="${1:-all}"
API_PROJECT="${API_PROJECT:-frs-api}"
WEB_PROJECT="${WEB_PROJECT:-frs-web}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

load() { [ -f "$1" ] && { set -a; . "$1"; set +a; } || true; }
load .run/vercel-token.txt
load .run/qdrant.env

[ -n "${VERCEL_TOKEN:-}" ] || die "VERCEL_TOKEN is not set (put it in .run/vercel-token.txt)"
case "$WHAT" in all|api|web) ;; *) die "usage: $0 [all|api|web]" ;; esac
if [ "$WHAT" != "web" ]; then
  [ -n "${QDRANT_URL:-}" ] && [ -n "${QDRANT_API_KEY:-}" ] ||
    die "QDRANT_URL and QDRANT_API_KEY are required (free cluster at https://cloud.qdrant.io; put them in .run/qdrant.env)"
fi
command -v npx >/dev/null || die "Node.js (npx) is required"

vc() { npx --yes vercel@latest "$@" --token "$VERCEL_TOKEN"; }

# set_env <dir> <name> <value>: replaces the production value
set_env() {
  ( cd "$1"; vc env rm "$2" production --yes >/dev/null 2>&1 || true
    printf '%s' "$3" | vc env add "$2" production >/dev/null ) || die "could not set $2"
}

# deploy <dir> <project>: links the project (creating it if new), deploys to production, prints its URL
deploy() {
  ( cd "$1"; vc link --yes --project "$2" >&2
    vc deploy --prod --yes ) | tail -n 1
}

# the project's real *.vercel.app address (the name may have been taken, so ask Vercel instead of guessing)
prod_url() {
  local host
  host="$(curl -fsS -H "Authorization: Bearer $VERCEL_TOKEN" "https://api.vercel.com/v9/projects/$1/domains" 2>/dev/null |
    python3 -c 'import sys,json;d=[x["name"] for x in json.load(sys.stdin)["domains"] if x["name"].endswith(".vercel.app")];print(d[0] if d else "")' 2>/dev/null)" || host=""
  if [ -n "$host" ]; then echo "https://$host"; else echo "$2"; fi
}

API_URL=""; WEB_URL=""

if [ "$WHAT" != "web" ]; then
  say "API: $API_PROJECT (backend/frs)"
  ( cd backend/frs; vc link --yes --project "$API_PROJECT" >&2 )
  set_env backend/frs QDRANT_URL "$QDRANT_URL"
  set_env backend/frs QDRANT_API_KEY "$QDRANT_API_KEY"
  set_env backend/frs FRS_CORS_ORIGINS "${FRS_CORS_ORIGINS:-*}"
  dep="$(deploy backend/frs "$API_PROJECT")"
  API_URL="$(prod_url "$API_PROJECT" "$dep")"
  say "API deployed: $API_URL"
  curl -fsS --max-time 60 "$API_URL/api/v1/health" && echo || echo "(health check failed: check the function logs in the Vercel dashboard)"
fi

if [ "$WHAT" != "api" ]; then
  [ -n "$API_URL" ] || API_URL="${NEXT_PUBLIC_API_BASE_URL:-https://$API_PROJECT.vercel.app}"
  say "Web: $WEB_PROJECT (frontend/frs_react), API at $API_URL"
  ( cd frontend/frs_react; vc link --yes --project "$WEB_PROJECT" >&2 )
  set_env frontend/frs_react NEXT_PUBLIC_API_BASE_URL "$API_URL"
  dep="$(deploy frontend/frs_react "$WEB_PROJECT")"
  WEB_URL="$(prod_url "$WEB_PROJECT" "$dep")"
  say "Web deployed: $WEB_URL"
fi

if [ "$WHAT" = "all" ]; then
  say "Locking API CORS to $WEB_URL"
  set_env backend/frs FRS_CORS_ORIGINS "$WEB_URL"
  deploy backend/frs "$API_PROJECT" >/dev/null
fi

echo
say "Done"
[ -n "$WEB_URL" ] && echo "  Live demo: $WEB_URL"
[ -n "$API_URL" ] && echo "  API:       $API_URL/api/v1/health"
echo "  Revoke the token when finished: https://vercel.com/account/tokens"

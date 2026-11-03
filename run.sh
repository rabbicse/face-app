#!/usr/bin/env bash
# One command to run the whole face sign-in stack on this machine, WITHOUT Docker.
#
#   ./run.sh               API + vector database + web app, then opens the browser (Ctrl+C stops everything)
#   ./run.sh --prod        same, with an optimised production build of the web app
#   ./run.sh help          all commands and options
#
# Needs Python 3.11+ and Node.js 22.12+. The vector database (Qdrant) runs inside the API process in
# local mode, so no database server is needed; registered faces are kept in .run/qdrant.
# To run with Docker instead, use ./docker.sh.
# Free ports are picked automatically (on a Mac, AirPlay Receiver usually holds port 5000).
# Works with the bash 3.2 that ships with macOS, and on Linux.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT/.run"
DB_DIR="$STATE_DIR/qdrant"
BACKEND="$ROOT/backend/frs"
FRONTEND="$ROOT/frontend/frs_react"
MOBILE="$ROOT/mobile/face_app"
cd "$ROOT"

# ---------------------------------------------------------------- output helpers
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi
step() { printf '%s==>%s %s\n' "$BLUE$BOLD" "$RESET" "$*"; }
ok()   { printf '%s ✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s !%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%s ✗ %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<EOF
${BOLD}Face sign-in: run the full stack locally, without Docker${RESET}

${BOLD}Usage:${RESET} ./run.sh [command] [options]

${BOLD}Commands${RESET}
  start      (default) Start the API (with its vector database) and the web app; Ctrl+C stops both
  reset      Delete all registered faces (asks first; -y to skip)
  test       Run the same checks as CI (backend, web, and mobile if Flutter is installed)
  mobile     Run the Flutter app on a connected phone/emulator against the running API
  help       Show this help

${BOLD}Options${RESET}
  --prod       Production build of the web app (faster pages, no hot reload; the build takes ~1 minute)
  --memory     Keep the database in memory only: faces are forgotten when you stop
  --no-blink   Turn off the blink check at sign-in (demos, or a camera that can't see blinks)
  --no-open    Don't open the browser
  -y, --yes    Don't ask for confirmation (reset)

${BOLD}Environment${RESET} (also read from .env)
  FRS_WEB_PORT (3000)  FRS_API_PORT (5000)  FRS_MATCH_THRESHOLD  FRS_MATCH_MARGIN
  QDRANT_URL, QDRANT_API_KEY   use a remote Qdrant (e.g. a free Qdrant Cloud cluster) instead of .run/qdrant
  A port that is set explicitly is used as is; otherwise the next free port is chosen.

${BOLD}Examples${RESET}
  ./run.sh                    # first run installs packages and downloads the face models
  ./run.sh --prod --no-blink  # fast demo
  ./run.sh test
  ./docker.sh                 # the same stack in Docker instead
EOF
}

# ---------------------------------------------------------------- arguments
COMMAND="start"
PROD=0; MEMORY=0; NO_BLINK=0; OPEN_BROWSER=1; ASSUME_YES=0; EXTRA=()
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then COMMAND="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --prod) PROD=1 ;;
        --memory) MEMORY=1 ;;
        --no-blink) NO_BLINK=1 ;;
        --no-open) OPEN_BROWSER=0 ;;
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help) COMMAND="help" ;;
        -*) die "Unknown option: $1 (see ./run.sh help)" ;;
        *) EXTRA+=("$1") ;;
    esac
    shift
done

# ---------------------------------------------------------------- helpers
# Value of KEY from the environment, else from .env, else the default.
setting() {
    local key="$1" default="$2" val=""
    if [ -n "${!key+x}" ]; then printf '%s' "${!key}"; return; fi
    if [ -f "$ROOT/.env" ]; then
        val="$(grep -E "^[[:space:]]*${key}=" "$ROOT/.env" | tail -n1 | cut -d= -f2- | sed -e 's/[[:space:]]*#.*$//' -e 's/^["'\'']//' -e 's/["'\'']$//' || true)"
    fi
    printf '%s' "${val:-$default}"
}
is_explicit() { [ -n "${!1+x}" ] || { [ -f "$ROOT/.env" ] && grep -qE "^[[:space:]]*$1=[^[:space:]#]" "$ROOT/.env"; }; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

# First free port starting at $1 (tries 30). Prints nothing if none is free.
free_port() {
    local p="$1" i=0
    while [ $i -lt 30 ]; do
        if ! port_busy "$p"; then printf '%s' "$p"; return; fi
        p=$((p + 1)); i=$((i + 1))
    done
}

# Port from FRS_*_PORT if set, else the first free one from the default.
choose_port() { # VAR default label
    local p
    if is_explicit "$1"; then
        p="$(setting "$1" "$2")"
        port_busy "$p" && die "Port $p ($1) is in use. Free it or choose another port."
    else
        p="$(free_port "$2")"
        [ -n "$p" ] || die "No free port found from $2 upwards. Set $1."
        [ "$p" = "$2" ] || warn "Port $2 is in use$( [ "$2" = 5000 ] && echo " (on a Mac usually AirPlay Receiver)"), so the $3 uses $p."
    fi
    printf '%s' "$p"
}

http_ok() { curl -fsS -o /dev/null --max-time 3 "$1" 2>/dev/null; }

wait_for() { # url seconds label
    local url="$1" limit="$2" label="$3" waited=0
    printf '%s    waiting for %s%s\n' "$DIM" "$label" "$RESET"
    until http_ok "$url"; do
        [ $waited -ge "$limit" ] && return 1
        sleep 2; waited=$((waited + 2))
    done
}

open_url() {
    [ "$OPEN_BROWSER" = 1 ] || return 0
    if have open; then open "$1" >/dev/null 2>&1 || true
    elif have xdg-open; then xdg-open "$1" >/dev/null 2>&1 || true
    fi
}

lan_ip() {
    local ip=""
    if have ipconfig; then ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"; fi
    if [ -z "$ip" ] && have hostname; then ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"; fi
    printf '%s' "${ip:-<your-lan-ip>}"
}

version_ge() { # "22.12.0" "22.12.0" -> true if $1 >= $2
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -n1)" = "$2" ]
}

# Every descendant PID of $1 (collected before anything is killed, so none get re-parented and missed).
descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null || true); do
        printf '%s\n' "$child"
        descendants "$child"
    done
}
# Ask PIDs to stop; whatever is still running 3 s later (npm can ignore SIGTERM) is killed.
stop_pids() {
    [ -n "$*" ] || return 0
    kill -TERM "$@" 2>/dev/null || true
    local i=0 alive
    while [ $i -lt 6 ]; do
        alive=""
        for p in "$@"; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
        [ -z "$alive" ] && return 0
        sleep 0.5; i=$((i + 1))
    done
    kill -KILL $alive 2>/dev/null || true
}
kill_tree() {
    stop_pids $(descendants "$1")
}

# Runs in the background: remembers the servers started by PID $1 and stops them if $1 disappears without
# cleaning up. Once $1 is gone its children are re-parented, so they must be known beforehand.
watchdog() {
    local parent="$1" known=""
    trap '' INT HUP # Ctrl+C is the main script's job
    while kill -0 "$parent" 2>/dev/null; do
        known="$(descendants "$parent")"
        sleep 2
    done
    stop_pids $known
    exit 0
}

blink_value() { [ "$NO_BLINK" = 1 ] && echo false || echo true; }

# ---------------------------------------------------------------- toolchains
ensure_python_env() { # requirements file
    local py=""
    for c in python3.13 python3.12 python3.11 python3; do
        if have "$c" && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then py="$c"; break; fi
    done
    [ -n "$py" ] || die "Python 3.11+ is required: https://www.python.org/downloads/"

    local venv="$BACKEND/.venv" req="$1" stamp hash
    if [ ! -x "$venv/bin/python" ]; then
        step "Creating a Python environment in backend/frs/.venv"
        "$py" -m venv "$venv"
    fi
    stamp="$venv/.installed-$req"
    hash="$(cat "$BACKEND/$req" "$BACKEND/requirements-api.txt" | cksum | awk '{print $1}')"
    if [ "$(cat "$stamp" 2>/dev/null || true)" != "$hash" ]; then
        step "Installing Python packages ($req)"
        "$venv/bin/python" -m pip install --quiet --disable-pip-version-check -r "$BACKEND/$req"
        printf '%s' "$hash" > "$stamp"
    fi
    PYTHON="$venv/bin/python"
}

ensure_node_env() {
    have node || die "Node.js 22.12+ is required: https://nodejs.org"
    local v; v="$(node -p 'process.versions.node')"
    version_ge "$v" "22.12.0" || die "Node.js 22.12+ is required, found $v."
    if [ ! -d "$FRONTEND/node_modules" ] || [ "$FRONTEND/package-lock.json" -nt "$FRONTEND/node_modules/.package-lock.json" ]; then
        step "Installing web packages (npm ci)"
        (cd "$FRONTEND" && npm ci --no-audit --no-fund)
    fi
}

# ---------------------------------------------------------------- commands
cmd_start() {
    have curl || die "curl is required."
    ensure_python_env requirements-api.txt
    ensure_node_env
    mkdir -p "$STATE_DIR"

    local api_port web_port db db_label qdrant_url
    api_port="$(choose_port FRS_API_PORT 5000 API)"
    web_port="$(choose_port FRS_WEB_PORT 3000 "web app")"
    qdrant_url="$(setting QDRANT_URL "")"
    if [ -n "$qdrant_url" ]; then
        # remote Qdrant (e.g. Qdrant Cloud): nothing runs locally for the database
        db=""; db_label="remote Qdrant at $qdrant_url"
        [ "$MEMORY" = 1 ] && warn "--memory is ignored because QDRANT_URL is set."
        export QDRANT_URL="$qdrant_url" QDRANT_API_KEY="$(setting QDRANT_API_KEY "")"
    elif [ "$MEMORY" = 1 ]; then db=":memory:"; db_label="in memory, forgotten on stop"
    else db="$DB_DIR"; mkdir -p "$db"; db_label="${DB_DIR#$ROOT/}  (./run.sh reset deletes them)"
    fi

    # Ctrl+C, closing the terminal or the script being stopped ends everything this script started, and
    # nothing else. If the script is killed outright (SIGKILL, no trap runs), the watchdog does it instead.
    trap 'trap - INT TERM HUP EXIT; printf "\n"; step "Stopping"; kill_tree $$; wait 2>/dev/null; exit 0' INT TERM HUP EXIT
    watchdog $$ &

    step "Starting the API on port $api_port"
    (
        cd "$BACKEND"
        # one process: Qdrant's local mode allows a single process to open the database folder.
        # QDRANT_LOCATION wins over QDRANT_URL in the API, so it is only set for a local database.
        if [ -n "$db" ]; then export QDRANT_LOCATION="$db"; else unset QDRANT_LOCATION; fi
        FRS_CORS_ORIGINS="$(setting FRS_CORS_ORIGINS "*")" \
        FRS_MATCH_THRESHOLD="$(setting FRS_MATCH_THRESHOLD 0.45)" FRS_MATCH_MARGIN="$(setting FRS_MATCH_MARGIN 0.05)" \
            exec "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$api_port" 2>&1
    ) | sed -u "s/^/${BLUE}[api]${RESET} /" &
    wait_for "http://localhost:$api_port/api/v1/health" 60 "the API" || die "The API didn't start (see the [api] lines above)."
    ok "API ready"

    # Same-origin proxy, exactly like the Docker image: the browser calls /api/v1 on the web port,
    # and Next.js forwards it to the API (no CORS, and cross-origin isolation keeps working).
    export FRS_API_PROXY_URL="http://127.0.0.1:$api_port" NEXT_PUBLIC_API_BASE_URL="" \
        NEXT_PUBLIC_REQUIRE_BLINK="$(blink_value)" NEXT_TELEMETRY_DISABLED=1

    if [ "$PROD" = 1 ]; then
        step "Building the web app for production (about a minute; the first build downloads the face models)"
        (cd "$FRONTEND" && npm run build 2>&1 | sed -u "s/^/${GREEN}[web]${RESET} /") || die "The web build failed."
        step "Starting the web app on port $web_port"
        ( cd "$FRONTEND" && exec npm run start -- --port "$web_port" 2>&1 ) | sed -u "s/^/${GREEN}[web]${RESET} /" &
    else
        step "Starting the web dev server on port $web_port (the first start downloads the face models)"
        ( cd "$FRONTEND" && exec npm run dev -- --port "$web_port" 2>&1 ) | sed -u "s/^/${GREEN}[web]${RESET} /" &
    fi

    if wait_for "http://localhost:$web_port/api/v1/health" 300 "the web app"; then
        printf '\n%s Everything is running%s  (Ctrl+C stops it)\n\n' "$GREEN$BOLD" "$RESET"
        printf '  %-12s %s\n' "Web app" "http://localhost:$web_port"
        printf '  %-12s %s\n' "Sign in" "http://localhost:$web_port/face/login"
        printf '  %-12s %s\n' "Examples" "http://localhost:$web_port/examples"
        printf '  %-12s %s\n' "API docs" "http://localhost:$api_port/docs"
        printf '  %-12s %s\n' "Phone / LAN" "http://$(lan_ip):$api_port  (./run.sh mobile)"
        printf '  %-12s %s\n' "Mode" "$( [ "$PROD" = 1 ] && echo production || echo "development (hot reload)")"
        printf '  %-12s %s\n' "Blink check" "$( [ "$NO_BLINK" = 1 ] && echo off || echo on)"
        printf '  %-12s %s\n\n' "Faces" "$db_label"
        open_url "http://localhost:$web_port"
    else
        warn "The web app isn't answering yet; watch the [web] lines above."
    fi
    wait
}

cmd_reset() {
    [ -n "$(setting QDRANT_URL "")" ] && warn "QDRANT_URL is set: this only deletes the local database, not the remote one."
    if [ ! -d "$DB_DIR" ]; then ok "No saved faces to delete."; return; fi
    if [ "$ASSUME_YES" != 1 ]; then
        printf '%sThis deletes ALL faces registered with ./run.sh. Continue? [y/N] %s' "$YELLOW" "$RESET"
        local answer; read -r answer
        case "$answer" in y|Y|yes|YES) ;; *) echo "Cancelled."; exit 0 ;; esac
    fi
    rm -rf "$DB_DIR"
    ok "Deleted all registered faces (stop and start ./run.sh if it is running)"
}

cmd_test() {
    local failed=0
    step "Backend tests (pytest)"
    ensure_python_env requirements-dev.txt
    (cd "$BACKEND" && "$PYTHON" -m pytest -q) || failed=1

    step "Web tests, lint and type check"
    ensure_node_env
    (cd "$FRONTEND" && npm test && npm run lint && npx tsc --noEmit) || failed=1

    if have flutter; then
        step "Mobile analyze and tests"
        (cd "$MOBILE" && flutter analyze && flutter test) || failed=1
    else
        warn "Flutter isn't installed, so the mobile checks were skipped."
    fi
    [ $failed = 0 ] && ok "All checks passed" || die "Some checks failed (see above)."
}

cmd_mobile() {
    have flutter || die "Flutter isn't installed: https://docs.flutter.dev/get-started/install"
    have curl || die "curl is required."
    local api_port="" p
    if is_explicit FRS_API_PORT; then api_port="$(setting FRS_API_PORT 5000)"
    else
        for p in 5000 5001 5002 5003 5004 5005; do
            if http_ok "http://localhost:$p/api/v1/health"; then api_port="$p"; break; fi
        done
    fi
    [ -n "$api_port" ] && http_ok "http://localhost:$api_port/api/v1/health" \
        || die "The API isn't running. Start it first in another terminal with ./run.sh (or set FRS_API_PORT)."
    [ -f "$MOBILE/assets/models/w600k_mbf.onnx" ] || { step "Downloading the mobile face model"; (cd "$MOBILE" && tool/setup_models.sh); }
    local url="http://$(lan_ip):$api_port"
    step "Running the app against $url (the phone must be on the same network)"
    cd "$MOBILE"
    flutter pub get
    exec flutter run --dart-define=FRS_API="$url" --dart-define=REQUIRE_BLINK="$(blink_value)" ${EXTRA[@]+"${EXTRA[@]}"}
}

# ---------------------------------------------------------------- dispatch
case "$COMMAND" in
    start|up|dev) cmd_start ;;
    reset) cmd_reset ;;
    test) cmd_test ;;
    mobile) cmd_mobile ;;
    help) usage ;;
    *) usage; die "Unknown command: $COMMAND" ;;
esac

#!/usr/bin/env bash
# One command to run the whole face sign-in stack (Qdrant + API + web app) in Docker.
#
#   ./docker.sh              build if needed, start, wait until healthy, open the browser
#   ./docker.sh help         all commands and options
#
# Needs Docker with Compose v2. To run without Docker, use ./docker.sh.
# Free ports are picked automatically (on a Mac, AirPlay Receiver usually holds port 5000).
# Works with the bash 3.2 that ships with macOS, and on Linux.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT/.run"
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
${BOLD}Face sign-in: run the full stack in Docker${RESET}

${BOLD}Usage:${RESET} ./docker.sh [command] [options]

${BOLD}Commands${RESET}
  start      (default) Build and start Qdrant, the API and the web app in Docker
  stop       Stop the Docker stack (registered faces are kept)
  restart    Stop, then start
  status     Show containers, ports and health
  logs       Follow the logs (optionally of one service: qdrant, api, web)
  reset      Stop and DELETE all registered faces (asks first; -y to skip)
  help       Show this help

${BOLD}Options${RESET}
  --no-blink   Turn off the blink check at sign-in (demos, or a camera that can't see blinks)
  --rebuild    Rebuild the Docker images from source
  --no-open    Don't open the browser
  -y, --yes    Don't ask for confirmation (reset)

${BOLD}Environment${RESET} (also read from .env, see .env.example)
  FRS_WEB_PORT (3000)  FRS_API_PORT (5000)  FRS_IMAGE_NAMESPACE  FRS_TAG  FRS_WORKERS
  FRS_MATCH_THRESHOLD  FRS_MATCH_MARGIN  FRS_CORS_ORIGINS
  A port that is set explicitly is used as is; otherwise the next free port is chosen.

${BOLD}Examples${RESET}
  ./docker.sh                    # first run builds the images (a few minutes), then opens http://localhost:3000
  ./docker.sh start --no-blink   # demo mode
  ./docker.sh logs api
  ./run.sh                    # the same stack without Docker
EOF
}

# ---------------------------------------------------------------- arguments
COMMAND="start"
NO_BLINK=0; REBUILD=0; OPEN_BROWSER=1; ASSUME_YES=0; EXTRA=()
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then COMMAND="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --no-blink) NO_BLINK=1 ;;
        --rebuild|--build) REBUILD=1 ;;
        --no-open) OPEN_BROWSER=0 ;;
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help) COMMAND="help" ;;
        -*) die "Unknown option: $1 (see ./docker.sh help)" ;;
        *) EXTRA+=("$1") ;;
    esac
    shift
done

# ---------------------------------------------------------------- shared helpers
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

http_ok() { curl -fsS -o /dev/null --max-time 3 "$1" 2>/dev/null; }

wait_for() { # url seconds label
    local url="$1" limit="$2" label="$3" waited=0
    printf '%s    waiting for %s%s' "$DIM" "$label" "$RESET"
    until http_ok "$url"; do
        if [ $waited -ge "$limit" ]; then printf '\n'; return 1; fi
        printf '.'; sleep 2; waited=$((waited + 2))
    done
    printf '\n'
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

# ---------------------------------------------------------------- docker helpers
compose() { docker compose "$@"; }

ensure_docker() {
    have docker || die "Docker is not installed. Get it from https://docs.docker.com/get-docker/, or run without Docker: ./run.sh"
    docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing. Update Docker Desktop, or install the compose plugin."
    if docker info >/dev/null 2>&1; then return; fi
    if [ "$(uname -s)" = "Darwin" ] && [ -d "/Applications/Docker.app" ]; then
        step "Starting Docker Desktop"
        open -a Docker
        local waited=0
        until docker info >/dev/null 2>&1; do
            [ $waited -ge 120 ] && die "Docker Desktop did not start within 2 minutes. Start it by hand and run this again."
            sleep 3; waited=$((waited + 3))
        done
        ok "Docker is running"
    else
        die "The Docker daemon isn't running. Start Docker (e.g. 'sudo systemctl start docker') and run this again."
    fi
}

# Published host port of a running service, or empty.
running_port() { compose port "$1" "$2" 2>/dev/null | head -n1 | awk -F: '{print $NF}' || true; }

pick_ports() {
    local cur_web cur_api
    cur_web="$(running_port web 3000)"; cur_api="$(running_port api 8000)"

    if [ -n "$cur_web" ] && ! is_explicit FRS_WEB_PORT; then WEB_PORT="$cur_web"
    elif is_explicit FRS_WEB_PORT; then WEB_PORT="$(setting FRS_WEB_PORT 3000)"
    else WEB_PORT="$(free_port 3000)"; [ -n "$WEB_PORT" ] || die "No free port found from 3000 upwards. Set FRS_WEB_PORT."
    fi

    if [ -n "$cur_api" ] && ! is_explicit FRS_API_PORT; then API_PORT="$cur_api"
    elif is_explicit FRS_API_PORT; then API_PORT="$(setting FRS_API_PORT 5000)"
    else API_PORT="$(free_port 5000)"; [ -n "$API_PORT" ] || die "No free port found from 5000 upwards. Set FRS_API_PORT."
    fi

    if [ "$API_PORT" != "5000" ] && ! is_explicit FRS_API_PORT && [ -z "$cur_api" ]; then
        warn "Port 5000 is in use (on a Mac usually AirPlay Receiver), so the API uses $API_PORT."
    fi
    if [ "$WEB_PORT" != "3000" ] && ! is_explicit FRS_WEB_PORT && [ -z "$cur_web" ]; then
        warn "Port 3000 is in use, so the web app uses $WEB_PORT."
    fi
    export FRS_WEB_PORT="$WEB_PORT" FRS_API_PORT="$API_PORT"
    if ! is_explicit FRS_CORS_ORIGINS; then export FRS_CORS_ORIGINS="http://localhost:$WEB_PORT"; fi
}

cmd_start() {
    ensure_docker
    have curl || die "curl is required."
    mkdir -p "$STATE_DIR"
    pick_ports

    local blink="true" namespace marker="$STATE_DIR/web-image-blink"
    [ "$NO_BLINK" = 1 ] && blink="false"
    namespace="$(setting FRS_IMAGE_NAMESPACE "")"

    if [ -n "$namespace" ]; then
        step "Pulling published images from Docker Hub ($namespace)"
        [ "$NO_BLINK" = 1 ] && warn "--no-blink needs a local build; the published web image keeps the blink check."
        compose pull
    else
        # The blink setting is baked into the web image at build time, so rebuild when it changes.
        local web_build=0 api_build=0
        docker image inspect "frs-web:$(setting FRS_TAG latest)" >/dev/null 2>&1 || web_build=1
        docker image inspect "frs-api:$(setting FRS_TAG latest)" >/dev/null 2>&1 || api_build=1
        [ "$(cat "$marker" 2>/dev/null || true)" = "$blink" ] || web_build=1
        [ "$REBUILD" = 1 ] && { web_build=1; api_build=1; }

        if [ $api_build = 1 ]; then
            step "Building the API image"
            compose build api
        fi
        if [ $web_build = 1 ]; then
            step "Building the web image (blink check: $blink). The first build downloads the face models and takes a few minutes"
            compose build --build-arg "NEXT_PUBLIC_REQUIRE_BLINK=$blink" web
            printf '%s' "$blink" > "$marker"
        fi
    fi

    step "Starting Qdrant, the API and the web app"
    compose up -d --no-build --remove-orphans

    local web="http://localhost:$WEB_PORT" api="http://localhost:$API_PORT"
    if ! wait_for "$api/api/v1/health" 120 "the API"; then
        compose ps; die "The API didn't become healthy. See: ./docker.sh logs api"
    fi
    if ! wait_for "$web/api/v1/health" 120 "the web app"; then
        compose ps; die "The web app didn't respond. See: ./docker.sh logs web"
    fi

    printf '\n%s Everything is running%s\n\n' "$GREEN$BOLD" "$RESET"
    printf '  %-14s %s\n' "Web app" "$web"
    printf '  %-14s %s\n' "Sign in" "$web/face/login"
    printf '  %-14s %s\n' "Examples" "$web/examples"
    printf '  %-14s %s\n' "API docs" "$api/docs"
    printf '  %-14s %s\n' "Phone / LAN" "http://$(lan_ip):$API_PORT  (mobile: ./run.sh mobile)"
    printf '  %-14s %s\n' "Blink check" "$( [ -n "$namespace" ] && echo "on (published image)" || cat "$marker")"
    printf '\n  %sLogs: ./docker.sh logs    Stop: ./docker.sh stop    Delete all faces: ./docker.sh reset%s\n\n' "$DIM" "$RESET"
    open_url "$web"
}

cmd_stop() {
    ensure_docker
    step "Stopping the stack (registered faces are kept)"
    compose down --remove-orphans
    ok "Stopped"
}

cmd_status() {
    ensure_docker
    compose ps
    local web api
    web="$(running_port web 3000)"; api="$(running_port api 8000)"
    printf '\n'
    if [ -n "$api" ] && http_ok "http://localhost:$api/api/v1/health"; then ok "API healthy        http://localhost:$api/docs"
    else warn "API not reachable"; fi
    if [ -n "$web" ] && http_ok "http://localhost:$web/api/v1/health"; then ok "Web app healthy    http://localhost:$web"
    else warn "Web app not reachable"; fi
}

cmd_logs() {
    ensure_docker
    compose logs -f --tail=100 ${EXTRA[@]+"${EXTRA[@]}"}
}

cmd_reset() {
    ensure_docker
    if [ "$ASSUME_YES" != 1 ]; then
        printf '%sThis deletes ALL registered faces. Continue? [y/N] %s' "$YELLOW" "$RESET"
        local answer; read -r answer
        case "$answer" in y|Y|yes|YES) ;; *) echo "Cancelled."; exit 0 ;; esac
    fi
    compose down -v --remove-orphans
    ok "Stopped and deleted the database volume"
}

# ---------------------------------------------------------------- dispatch
case "$COMMAND" in
    start|up) cmd_start ;;
    stop|down) cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status|ps) cmd_status ;;
    logs) cmd_logs ;;
    reset) cmd_reset ;;
    help) usage ;;
    *) usage; die "Unknown command: $COMMAND" ;;
esac

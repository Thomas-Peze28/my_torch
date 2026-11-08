#!/usr/bin/env bash
# Shared helpers of the demo scripts: go to the repository root, use the
# .venv if there is one, and show each command before running it.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 84
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

BOLD=$'\e[1m'
BLUE=$'\e[34m'
GREEN=$'\e[32m'
RED=$'\e[31m'
RESET=$'\e[0m'

title() {
    echo
    echo "${BOLD}${BLUE}=== $* ===${RESET}"
    echo
}

pause() {
    echo
    read -r -p "${BOLD}[Enter]${RESET} " _
}

run() {
    echo "${BOLD}\$ $*${RESET}"
    "$@"
    local code=$?
    echo "${BOLD}(exit code: $code)${RESET}"
    return 0
}

draw_board() {
    python3 - "$1" <<'PY'
import sys
pieces, side = sys.argv[1].split()[:2]
for rank, row in zip("87654321", pieces.split("/")):
    squares = "".join("." * int(c) if c.isdigit() else c for c in row)
    print(f"  {rank} " + " ".join(squares))
print("    a b c d e f g h")
print(f"  {'White' if side == 'w' else 'Black'} to move  (UPPERCASE = White, lowercase = Black)")
PY
}

need_file() {
    if [ ! -f "$1" ]; then
        echo "${RED}Missing $1: $2${RESET}" >&2
        exit 84
    fi
}

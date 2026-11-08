#!/usr/bin/env bash
# Runs every demo in order, with a pause between them.

DEMO="$(dirname "${BASH_SOURCE[0]}")"
for script in "$DEMO"/[0-9]_*.sh; do
    bash "$script"
    echo
    read -r -p $'\e[1m[Enter] next demo\e[0m ' _
done

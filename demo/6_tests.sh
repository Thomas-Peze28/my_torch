#!/usr/bin/env bash
# Demo 6: the unit tests of the library, the analyzer and the scripts.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

title "Unit tests"
run python3 -m pytest tests -q

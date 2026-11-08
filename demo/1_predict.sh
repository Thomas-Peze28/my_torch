#!/usr/bin/env bash
# Demo 1: the pushed network answers on chessboards it has never seen.
# Usage: demo/1_predict.sh [FILE] [COUNT]
# Shows the first COUNT chessboards (10 by default) of FILE (the test file
# by default).

FILE="${1:-dataset/test.txt}"
COUNT="${2:-10}"
if [ -n "$1" ]; then
    FILE="$(realpath -m "$1")"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BOARDS="$(mktemp)"
trap 'rm -f "$BOARDS" "$BOARDS.answers"' EXIT
need_file "$FILE" "no such chessboard file"
grep -v '^[[:space:]]*$' "$FILE" | head -n "$COUNT" > "$BOARDS"

title "$(wc -l < "$BOARDS") chessboards the network has never seen"
number=1
while read -r line; do
    fen="$(echo "$line" | cut -d' ' -f1-6)"
    echo "${BOLD}Chessboard $number${RESET}: $fen"
    draw_board "$fen"
    echo
    number=$((number + 1))
done < "$BOARDS"
pause

title "Prediction"
echo "${BOLD}\$ ./my_torch_analyzer --predict my_torch_network.nn boards.txt${RESET}"
./my_torch_analyzer --predict my_torch_network.nn "$BOARDS" > "$BOARDS.answers" || exit 84
echo
printf "${BOLD}%-3s %-18s %-18s${RESET}\n" "#" "expected" "network"
number=1
right=0
while read -r line && read -r given <&3; do
    expected="$(echo "$line" | cut -d' ' -f7-)"
    if [ -z "$expected" ]; then
        mark=""
    elif [ "$expected" = "$given" ]; then
        mark="${GREEN}right${RESET}"
        right=$((right + 1))
    else
        mark="${RED}wrong${RESET}"
    fi
    printf "%-3s %-18s %-18s %s\n" "$number" "${expected:--}" "$given" "$mark"
    number=$((number + 1))
done < "$BOARDS" 3< "$BOARDS.answers"
echo
echo "${BOLD}$right right answers out of $((number - 1))${RESET}"

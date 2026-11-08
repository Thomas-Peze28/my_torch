#!/usr/bin/env bash
# Demo 2: the score of the pushed network on the 71,556 test chessboards,
# never used to train it nor to choose any setting.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_file dataset/test.txt "run 'make dataset' first"

title "Score on $(wc -l < dataset/test.txt) chessboards never seen"
echo "${BOLD}\$ time ./my_torch_analyzer --predict my_torch_network.nn dataset/test.txt${RESET}"
time ./my_torch_analyzer --predict my_torch_network.nn dataset/test.txt > /dev/null
pause

title "Score for each answer"
run python3 -m scripts.evaluate_network my_torch_network.nn --file dataset/test.txt

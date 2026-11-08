#!/usr/bin/env bash
# Demo 4: the whole chain, live: create an untrained network from a
# configuration file, train it on 20,000 chessboards, then measure it.
# The files are written in runs/demo/ (not pushed).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_file dataset/train.txt "run 'make dataset' first"

rm -rf runs/demo
mkdir -p runs/demo
head -n 20000 dataset/train.txt > runs/demo/train.txt
head -n 5000 dataset/validation.txt > runs/demo/check.txt

title "1. The configuration: 769 inputs -> 64 neurons -> 5 answers"
cat configs/basic_network.conf
pause

title "2. Create an untrained network"
cd runs/demo || exit 84
run ../../my_torch_generator ../../configs/basic_network.conf 1
cd "$ROOT" || exit 84
echo
echo "${BOLD}Untrained, it answers at random:${RESET}"
./my_torch_analyzer --predict runs/demo/basic_network_1.nn runs/demo/check.txt | sort | uniq -c
pause

title "3. Train it on 20,000 chessboards"
run ./my_torch_analyzer --train --save runs/demo/trained.nn runs/demo/basic_network_1.nn runs/demo/train.txt
pause

title "4. Measure it on 5,000 chessboards it has never seen"
run python3 -m scripts.evaluate_network runs/demo/trained.nn --file runs/demo/check.txt
pause

title "5. The pushed network on the same 5,000 chessboards"
echo "Trained on 2,275,920 chessboards (with mirrors and swapped colours), with two wider hidden layers (769 -> 512 -> 256 -> 5)."
run python3 -m scripts.evaluate_network my_torch_network.nn --file runs/demo/check.txt

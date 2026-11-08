#!/usr/bin/env bash
# Demo 3: every wrong use writes a message on the error output and exits
# with code 84. The broken files are in demo/files/.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

title "Wrong arguments"
run ./my_torch_analyzer
run ./my_torch_analyzer --predict --train my_torch_network.nn demo/files/bad_fen.txt
run ./my_torch_analyzer --predict --save out.nn my_torch_network.nn demo/files/bad_fen.txt
pause

title "Wrong files"
run ./my_torch_analyzer --predict missing.nn demo/files/bad_fen.txt
run ./my_torch_analyzer --predict demo/files/broken.nn demo/files/bad_fen.txt
run ./my_torch_analyzer --predict my_torch_network.nn demo/files/bad_fen.txt
pause

title "Training files need a known answer"
run ./my_torch_analyzer --train --save out.nn my_torch_network.nn demo/files/bad_answer.txt
run ./my_torch_analyzer --train --save out.nn my_torch_network.nn demo/files/no_answer.txt

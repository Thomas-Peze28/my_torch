#!/usr/bin/env bash
# Demo 5: the benchmark graphs that justify every setting.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

show() {
    title "$1"
    echo "$2"
    xdg-open "benchmarks/$3" > /dev/null 2>&1 &
    pause
}

show "Learning curves of the pushed network" \
     "Dashed = learned chessboards, solid = put aside. The dot is the kept epoch (early stopping)." \
     pushed_network.png
show "Mistakes of the pushed network" \
     "Rows = right answer, columns = answer of the network. Almost no colour mix-up (3 out of 71,556); check <-> checkmate is the hard part." \
     pushed_network_confusion.png
show "Starting weights" "uniform 61.9%, LeCun 78.7%, Xavier 78.2%, He 78.8%: He is kept." init.png
show "Optimizer" "SGD 78.8%, Adam 80.5%: Adam is kept." optimizer.png
show "Learning rate" "0.0001 too slow (71.4%), 0.1 too big (54.2%), 0.001 is kept (80.5%)." learning_rate.png
show "Batch size" "16: 81.2% in 2.3 min, 64: 80.5% in 1.5 min, 256: 79.5% in 0.8 min. 64 is a compromise." batch_size.png
show "Activation" "ReLU 80.5%, Sigmoid 77.1%: ReLU is kept." activation.png
show "Loss" "Cross-entropy 80.5%, MSE 80.7%: same accuracy, cross-entropy is made for choosing one answer." loss.png
show "More chessboards" "Same network, with mirrors then swapped colours: 83.3% -> 87.0% -> 91.4% on the test, and much less learning by heart." augmentation.png
show "Network size" "With four times more chessboards, a wider network (769 -> 512 -> 256 -> 5) gains 0.7 point: 92.1%." size.png

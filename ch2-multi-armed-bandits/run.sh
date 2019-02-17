#!/bin/bash

set -eux

rm -rf output_dir

python main.py --epsilon 0.00 ./output_dir/epsilon_0
python main.py --epsilon 0.01 ./output_dir/epsilon_01
python main.py --epsilon 0.1  ./output_dir/epsilon_10

tensorboard --logdir ./output_dir

#!/bin/bash

set -eux

rm -rf output_dir

STEPS=50
ARMS=10

python main.py --steps $STEPS \
               --arms $ARMS \
               --epsilon 0.1 \
               --init_q 0 \
               --optimizer recency_weighted ./output_dir/recency_weighted
python main.py --steps $STEPS \
               --arms $ARMS \
               --epsilon 0.1 \
               --init_q 5 \
               --optimizer recency_weighted ./output_dir/recency_weighted_optimistic

tensorboard --logdir ./output_dir

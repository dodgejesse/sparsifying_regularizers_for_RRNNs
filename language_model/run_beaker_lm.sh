#!/usr/bin/env bash

python3.6 -u language_model/train_lm.py \
       --train /data/train \
       --dev /data/dev \
       --test /data/test \
       --logging_dir /output/logging/ \
       --gpu 1 \
       --lr 1.0 \
       --lr_decay 0.98 \
       --lr_decay_epoch 150 \
       --activation "tanh" \
       --batch_size 32 \
       --model "rrnn" \
       --depth 2 \
       --dropout 0.2 \
       --rnn_dropout 0.2 \
       --use_output_gate True \
       --unroll_size 35 \
       --use_rho False \
       --max_epoch 350 \
       --weight_decay 1e-5 \
       --patience 30 \
       --sparsity_type none \
       --reg_strength 0.01

best_dev=$(cat /output/logging/layers=* | grep -oE 'dev_ppl=([0-9\.]*)' | cut -d "=" -f2 | sort -rn | tail -n 1)

echo '{ "best_dev": '$best_dev' }' > /output/metrics.json

#!/bin/bash

is_peft=False
num_gpus=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$num_gpus main.py --config-name=decision.yaml mode=test_decision \
    is_peft=$is_peft
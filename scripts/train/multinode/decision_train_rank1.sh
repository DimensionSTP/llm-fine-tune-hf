#!/bin/bash

num_machines=2
num_processes_per_machine=8
num_processes=$((num_machines * num_processes_per_machine))
machine_rank=1
main_process_ip="RANK0_NODE_IP"
main_process_port=29500
gpu_ids="0,1,2,3,4,5,6,7"

accelerate launch \
    --num_machines=$num_machines \
    --num_processes=$num_processes \
    --machine_rank=$machine_rank \
    --main_process_ip=$main_process_ip \
    --main_process_port=$main_process_port \
    --same_network \
    --gpu_ids=$gpu_ids \
    main.py --config-name=decision.yaml mode=train \
    devices=$num_processes_per_machine \
    distributed.enabled=true \
    distributed.num_machines=$num_machines \
    distributed.num_processes_per_machine=$num_processes_per_machine \
    distributed.machine_rank=$machine_rank \
    distributed.main_process_ip=$main_process_ip \
    distributed.main_process_port=$main_process_port \
    distributed.validation_mode=error
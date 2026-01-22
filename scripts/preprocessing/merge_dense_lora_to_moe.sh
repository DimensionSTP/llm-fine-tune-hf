#!bin/bash

path="src/preprocessing"

model_types=(
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-4B"
)

experts_combinations=(
    "8 2"
    "8 3"
    "8 4"
    "4 2"
    "4 3"
)

train_method="sft"

merge_modes=(
    "one2n"
    "n2n"
    "m2n"
)

merge_attention=True

one2n_base_adapter="full_split"
one2n_alphas=(1.0 0.95 0.90 0.85 0.80 0.75 0.70 0.65)

n2n_adapter_paths=("split_8_0" "split_8_1" "split_8_2" "split_8_3" "split_8_4" "split_8_5" "split_8_6" "split_8_7")
n2n_alphas=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

m2n_adapter_paths=("split_4_0" "split_4_1" "split_4_2" "split_4_3")
m2n_expert_to_adapter=(0 1 2 3 0 1 2 3)
m2n_group_alphas=(1.0 0.90)
m2n_per_expert_alphas=(1.0 1.0 1.0 1.0 0.90 0.90 0.90 0.90)
m2n_expert_to_group=(0 0 0 0 1 1 1 1)

for model_type in "${model_types[@]}"
do
    for combo in "${experts_combinations[@]}"
    do
        set -- $combo
        num_experts=$1
        num_experts_per_tok=$2

        for merge_mode in "${merge_modes[@]}"
        do
            echo "----------------------------------------------------------------"
            echo "Model type: $model_type"
            echo "Num experts: $num_experts"
            echo "Num experts per tok: $num_experts_per_tok"
            echo "Merge mode: $merge_mode"
            echo "Merge attention: $merge_attention"
            echo "----------------------------------------------------------------"

            python $path/merge_dense_lora_to_moe.py \
                model_type=$model_type \
                dense_to_moe.moe.num_experts=$num_experts \
                dense_to_moe.moe.num_experts_per_tok=$num_experts_per_tok \
                dense_to_moe.merge_mode=$merge_mode \
                dense_to_moe.targets.merge_attention=$merge_attention \
                dense_to_moe.one2n.base_adapter=$one2n_base_adapter \
                dense_to_moe.one2n.alphas=$one2n_alphas \
                dense_to_moe.n2n.adapter_paths=$n2n_adapter_paths \
                dense_to_moe.n2n.alphas=$n2n_alphas \
                dense_to_moe.m2n.adapter_paths=$m2n_adapter_paths \
                dense_to_moe.m2n.expert_to_adapter=$m2n_expert_to_adapter \
                dense_to_moe.m2n.group_alphas=$m2n_group_alphas \
                dense_to_moe.m2n.per_expert_alphas=$m2n_per_expert_alphas \
                dense_to_moe.m2n.expert_to_group=$m2n_expert_to_group

            echo "----------------------------------------------------------------"
            echo "Done merging dense lora to moe: $model_type-experts_$num_experts-tok_$num_experts_per_tok-merge_mode_$merge_mode-merge_attention_$merge_attention"
            echo "----------------------------------------------------------------"
        done
    done
done
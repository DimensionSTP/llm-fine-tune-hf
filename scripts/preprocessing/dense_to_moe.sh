#!bin/bash

path="src/preprocessing"

norm_topk_prob=true
output_router_logits=false
router_aux_loss_coef=0.001
init_type="zeros"
gain=0.02

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

for model_type in "${model_types[@]}"
do
    for combo in "${experts_combinations[@]}"
    do
        set -- $combo
        num_experts=$1
        num_experts_per_tok=$2
        
        echo "----------------------------------------------------------------"
        echo "Model type: $model_type"
        echo "Num experts: $num_experts"
        echo "Num experts per tok: $num_experts_per_tok"
        echo "----------------------------------------------------------------"

        python $path/dense_to_moe_qwen3.py \
            model_type=$model_type \
            dense_to_moe.moe.num_experts=$num_experts \
            dense_to_moe.moe.num_experts_per_tok=$num_experts_per_tok \
            dense_to_moe.moe.norm_topk_prob=$norm_topk_prob \
            dense_to_moe.moe.output_router_logits=$output_router_logits \
            dense_to_moe.moe.router_aux_loss_coef=$router_aux_loss_coef \
            dense_to_moe.router.init_type=$init_type \
            dense_to_moe.router.gain=$gain
        
        echo "----------------------------------------------------------------"
        echo "Done converting dense to moe: $model_type-experts_$num_experts-tok_$num_experts_per_tok"
        echo "----------------------------------------------------------------"
    done
done
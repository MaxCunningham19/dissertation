#!/bin/bash
scalarizations=("linear" "chebyshev")
normalizations=("L1" "Softmax")
model_path="$1"
env="$2"
max_steps="$3"


for scalarization in "${scalarizations[@]}"
do
    python calc_reward_differences.py \
        --env $env \
        --model democratic \
        --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
        --model_path $model_path \
        --max_steps $max_steps \
        --images_dir images/rewards
    
    python calc_reward_differences.py \
        --env $env \
        --model democratic_dwl \
        --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
        --model_path $model_path \
        --max_steps $max_steps \
        --images_dir images/rewards
    
    for normalization in "${normalizations[@]}"
    do
        python calc_reward_differences.py \
            --env $env \
            --model scaled_democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization \
            --model_path $model_path \
            --max_steps $max_steps \
            --images_dir images/rewards

        for wnorm in "${normalizations[@]}"
        do
            python calc_reward_differences.py \
                --env $env \
                --model scaled_democratic_dwl \
                --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization w_normalization=$wnorm \
                --model_path $model_path \
                --max_steps $max_steps \
                --images_dir images/rewards
        done
    done
done

for normalization in "${normalizations[@]}"
do
    python calc_reward_differences.py \
        --env $env \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy w_normalization=$normalization \
        --model_path $model_path \
        --max_steps $max_steps \
        --images_dir images/rewards
done

python plot_reward_differences.py \
    --env $env \
    --folder_path images/rewards/$env \
    --objective_labels $4
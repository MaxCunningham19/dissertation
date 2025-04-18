#!/bin/bash

preference_vectors=(
    "0.0 1.0"
    "0.1 0.9"
    "0.2 0.8"
    "0.3 0.7"
    "0.4 0.6"
    "0.5 0.5"
    "0.6 0.4"
    "0.7 0.3"
    "0.8 0.2"
    "0.9 0.1"
    "1.0 0.0"
)
scalarizations=("linear" "chebyshev")
normalizations=("L1" "Softmax")
collaborative_model_path="$1"

for prefs in "${preference_vectors[@]}"
do
    prefs_cleaned="${prefs// /_}"
    for scalarization in "${scalarizations[@]}"
    do
        python plot_agent_actions.py \
            --env mo-collaborative-env \
            --model democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $collaborative_model_path \
            --objective_labels food water \
            --action_labels noop U D L R LU RU LD RD \
            --human_preference $prefs \
            --images_dir images/collaborative/democratic/$prefs_cleaned/$scalarization
        
        for normalization in "${normalizations[@]}"
        do
            python plot_agent_actions.py \
                --env mo-collaborative-env \
                --model democratic_dwl \
                --model_kwargs hidlyr_nodes=128 scalarization=$scalarization w_normalization=$normalization \
                --model_path $collaborative_model_path \
                --objective_labels food water \
                --action_labels noop U D L R LU RU LD RD \
                --human_preference $prefs \
                --images_dir images/collaborative/democratic_dwl/$prefs_cleaned/$scalarization/w_$normalization

            python plot_agent_actions.py \
                --env mo-collaborative-env \
                --model scaled_democratic \
                --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization \
                 --model_path $collaborative_model_path \
                --objective_labels food water \
                --action_labels noop U D L R LU RU LD RD \
                --human_preference $prefs \
                --images_dir images/collaborative/scaled_democratic/$prefs_cleaned/$scalarization/$normalization

            for wnormalization in "${normalizations[@]}"
            do
                python plot_agent_actions.py \
                    --env mo-collaborative-env \
                    --model scaled_democratic_dwl \
                    --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization w_normalization=$wnormalization \
                     --model_path $collaborative_model_path \
                    --objective_labels food water \
                    --action_labels noop U D L R LU RU LD RD \
                    --human_preference $prefs \
                    --images_dir images/collaborative/scaled_democratic_dwl/$prefs_cleaned/$scalarization/$normalization/w_$wnormalization
            done
        done
    done
    python plot_agent_actions.py \
        --env mo-collaborative-env \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy \
        --model_path $collaborative_model_path \
        --objective_labels food water \
        --action_labels noop U D L R LU RU LD RD \
        --human_preference $prefs \
        --images_dir images/collaborative/dwl/$prefs_cleaned
done

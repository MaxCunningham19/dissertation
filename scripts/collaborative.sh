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
            --images_dir images/collaborative/democratic/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-collaborative-env \
            --model democratic_dwl \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization \
            --model_path $collaborative_model_path \
            --objective_labels food water \
            --action_labels noop U D L R LU RU LD RD \
            --human_preference $prefs \
            --images_dir images/collaborative/democratic_dwl/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-collaborative-env \
            --model scaled_democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $collaborative_model_path \
            --objective_labels food water \
            --action_labels noop U D L R LU RU LD RD \
            --human_preference $prefs \
            --images_dir images/collaborative/democratic/$scalarization/$prefs_cleaned
    done
    python plot_agent_actions.py \
        --env mo-collaborative-env \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy\
        --model_path $collaborative_model_path \
        --objective_labels food water \
        --action_labels noop U D L R LU RU LD RD \
        --human_preference $prefs \
        --images_dir images/collaborative/dwl/$prefs_cleaned
done

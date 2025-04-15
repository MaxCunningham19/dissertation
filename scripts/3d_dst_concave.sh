#!/bin/bash

preference_vectors=(
    "0.0 0.0 1.0"
    "0.0 0.1 0.9"
    "0.0 0.2 0.8"
    "0.0 0.3 0.7"
    "0.0 0.4 0.6"
    "0.0 0.6 0.4"
    "0.0 0.7 0.3"
    "0.0 0.8 0.2"
    "0.0 0.9 0.1"
    "0.0 1.0 0.0"
    "0.1 0.0 0.9"
    "0.1 0.2 0.7"
    "0.1 0.3 0.6"
    "0.1 0.4 0.5"
    "0.1 0.5 0.4"
    "0.1 0.6 0.3"
    "0.1 0.7 0.2"
    "0.1 0.9 0.0"
    "0.2 0.0 0.8"
    "0.2 0.1 0.7"
    "0.2 0.3 0.5"
    "0.2 0.5 0.3"
    "0.2 0.8 0.0"
    "0.3 0.0 0.7"
    "0.3 0.1 0.6"
    "0.3 0.2 0.5"
    "0.33 0.33 0.33"
    "0.3 0.5 0.2"
    "0.3 0.7 0.0"
    "0.4 0.0 0.6"
    "0.4 0.1 0.5"
    "0.4 0.5 0.1"
    "0.4 0.6 0.0"
    "0.5 0.1 0.4"
    "0.5 0.2 0.3"
    "0.5 0.3 0.2"
    "0.5 0.4 0.1"
    "0.6 0.0 0.4"
    "0.6 0.1 0.3"
    "0.6 0.4 0.0"
    "0.7 0.0 0.3"
    "0.7 0.1 0.2"
    "0.7 0.3 0.0"
    "0.8 0.0 0.2"
    "0.8 0.2 0.0"
    "0.9 0.0 0.1"
    "0.9 0.1 0.0"
    "1.0 0.0 0.0" 
)
scalarizations=("linear" "chebyshev")
dst_concave_model_path="$1"

for prefs in "${preference_vectors[@]}"
do
    prefs_cleaned="${prefs// /_}"
    for scalarization in "${scalarizations[@]}"
    do
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-concave-v0 \
            --model democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_concave_model_path \
            --objective_labels Treasure Speed Boarder  \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_concave/democratic/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-concave-v0 \
            --model democratic_dwl \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_concave_model_path \
            --objective_labels Treasure Speed Boarder  \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_concave/democratic_dwl/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-concave-v0 \
            --model scaled_democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_concave_model_path \
            --objective_labels Treasure Speed Boarder  \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_concave/scaled_democratic/$scalarization/$prefs_cleaned
    done
    python plot_agent_actions.py \
        --env mo-3d-deep-sea-treasure-concave-v0 \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy\
        --model_path $dst_concave_model_path \
        --objective_labels Treasure Speed Boarder \
        --action_labels U D L R \
        --human_preference $prefs \
        --images_dir images/dst/3d_concave/dwl/$prefs_cleaned
done

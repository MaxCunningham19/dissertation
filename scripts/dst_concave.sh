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
dst_concave_model_path="path"

for prefs in "${preference_vectors[@]}"
do
    prefs_cleaned="${prefs// /_}"
    for scalarization in "${scalarizations[@]}"
    do
        python plot_agent_actions.py \
            --env mo-deep-sea-treasure-concave-v0 \
            --model democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_concave_model_path \
            --objective_labels treasure speed \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/concave/democratic/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-deep-sea-treasure-concave-v0 \
            --model democratic_dwl \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization w_exploration_strategy=greedy\
            --model_path $dst_concave_model_path \
            --objective_labels treasure speed \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/concave/democratic_dwl/$scalarization/$prefs_cleaned
        
        python plot_agent_actions.py \
            --env mo-deep-sea-treasure-concave-v0 \
            --model scaled_democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_concave_model_path \
            --objective_labels treasure speed \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/concave/democratic/$scalarization/$prefs_cleaned
    done
    python plot_agent_actions.py \
        --env mo-deep-sea-treasure-concave-v0 \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy\
        --model_path $dst_concave_model_path \
        --objective_labels treasure speed \
        --action_labels U D L R \
        --human_preference $prefs \
        --images_dir images/dst/concave/dwl/$prefs_cleaned
done

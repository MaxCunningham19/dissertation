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
dst_convex_model_path="$1"

for prefs in "${preference_vectors[@]}"
do
    prefs_cleaned="${prefs// /_}"
    python plot_agent_actions.py \
        --env mo-deep-sea-treasure-convex-v0 \
        --model dwl \
        --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy \
        --model_path $dst_convex_model_path \
        --objective_labels treasure speed \
        --action_labels U D L R \
        --human_preference $prefs \
        --images_dir images/dst/convex/dwl/$prefs_cleaned
done

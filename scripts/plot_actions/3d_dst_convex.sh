#!/bin/bash

preference_vectors=(
    "0.0 0.0 1.0"
    "0.0 0.1 0.9"
    "0.0 0.2 0.8"
    "0.0 0.3 0.7"
    "0.0 0.4 0.6"
    "0.0 0.5 0.5"
    "0.0 0.6 0.4"
    "0.0 0.7 0.3"
    "0.0 0.8 0.2"
    "0.0 0.9 0.1"
    "0.0 1.0 0.0"
    "0.1 0.0 0.9"
    "0.1 0.1 0.8"
    "0.1 0.2 0.7"
    "0.1 0.3 0.6"
    "0.1 0.4 0.5"
    "0.1 0.5 0.4"
    "0.1 0.6 0.3"
    "0.1 0.7 0.2"
    "0.1 0.8 0.1"
    "0.1 0.9 0.0"
    "0.2 0.0 0.8"
    "0.2 0.1 0.7"
    "0.2 0.2 0.6"
    "0.2 0.3 0.5"
    "0.2 0.4 0.4"
    "0.2 0.5 0.3"
    "0.2 0.6 0.2"
    "0.2 0.7 0.1"
    "0.2 0.8 0.0"
    "0.3 0.0 0.7"
    "0.3 0.1 0.6"
    "0.3 0.2 0.5"
    "0.3 0.3 0.4"
    "0.3 0.4 0.3"
    "0.3 0.5 0.2"
    "0.3 0.6 0.1"
    "0.3 0.7 0.0"
    "0.33 0.33 0.33"
    "0.4 0.0 0.6"
    "0.4 0.1 0.5"
    "0.4 0.2 0.4"
    "0.4 0.3 0.3"
    "0.4 0.4 0.2"
    "0.4 0.5 0.1"
    "0.4 0.6 0.0"
    "0.5 0.0 0.5"
    "0.5 0.1 0.4"
    "0.5 0.2 0.3"
    "0.5 0.3 0.2"
    "0.5 0.4 0.1"
    "0.5 0.5 0.0"
    "0.6 0.0 0.4"
    "0.6 0.1 0.3"
    "0.6 0.2 0.2"
    "0.6 0.3 0.1"
    "0.6 0.4 0.0"
    "0.7 0.0 0.3"
    "0.7 0.1 0.2"
    "0.7 0.2 0.1"
    "0.7 0.3 0.0"
    "0.8 0.0 0.2"
    "0.8 0.1 0.1"
    "0.8 0.2 0.0"
    "0.9 0.0 0.1"
    "0.9 0.1 0.0"
    "1.0 0.0 0.0"
)
scalarizations=("linear" "chebyshev")
normalizations=("L1" "Softmax")
dst_convex_model_path="$1"

for prefs in "${preference_vectors[@]}"
do
    prefs_cleaned="${prefs// /_}"
    for scalarization in "${scalarizations[@]}"
    do
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-convex-v0 \
            --model democratic \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_convex_model_path \
            --objective_labels treasure speed penalty  \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_convex/democratic/$prefs_cleaned/$scalarization
        
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-convex-v0 \
            --model democratic_dwl \
            --model_kwargs hidlyr_nodes=128 scalarization=$scalarization\
            --model_path $dst_convex_model_path \
            --objective_labels treasure speed penalty  \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_convex/democratic_dwl/$prefs_cleaned/$scalarization
        
        for normalization in "${normalizations[@]}"
        do
            python plot_agent_actions.py \
                --env mo-3d-deep-sea-treasure-convex-v0 \
                --model scaled_democratic \
                --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization\
                --model_path $dst_convex_model_path \
                --objective_labels treasure speed penalty  \
                --action_labels U D L R \
                --human_preference $prefs \
                --images_dir images/dst/3d_convex/scaled_democratic/$prefs_cleaned/$scalarization/$normalization

            for wnorm in "${normalizations[@]}"
            do
                python plot_agent_actions.py \
                    --env mo-3d-deep-sea-treasure-convex-v0 \
                    --model scaled_democratic_dwl \
                    --model_kwargs hidlyr_nodes=128 scalarization=$scalarization normalization=$normalization w_normalization=$wnorm\
                    --model_path $dst_convex_model_path \
                    --objective_labels treasure speed penalty  \
                    --action_labels U D L R \
                    --human_preference $prefs \
                    --images_dir images/dst/3d_convex/scaled_democratic_dwl/$prefs_cleaned/$scalarization/$normalization/w_$wnorm
            done
        done
    done
    for normalization in "${normalizations[@]}"
    do
        python plot_agent_actions.py \
            --env mo-3d-deep-sea-treasure-convex-v0 \
            --model dwl \
            --model_kwargs hidlyr_nodes=128 w_exploration_strategy=greedy w_normalization=$normalization\
            --model_path $dst_convex_model_path \
            --objective_labels treasure speed penalty \
            --action_labels U D L R \
            --human_preference $prefs \
            --images_dir images/dst/3d_convex/dwl/$prefs_cleaned/w_$normalization
    done
done

#!/bin/bash

# Define the starting and ending index
start=0
end=7

# Loop through the desired range of indices
for i in $(seq $start $end); do
    echo "Running frame_to_multi_view.py with process-id: $i"
    CUDA_VISIBLE_DEVICES=$i python frame_to_multi_view.py --input-path data/real_benchmark_motion_160_seeds --checkpoint-path ckpts/V3D_512.ckpt --output-folder data/real_benchmark_motion_160_multi_view --num-steps 32 --process-id $i &
done

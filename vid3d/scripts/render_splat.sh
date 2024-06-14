#!/bin/bash

# Define the starting and ending index
start=0
end=7

# Loop through the desired range of indices
for i in $(seq $start $end); do
    echo "Running splat.py with process-id: $i"
    CUDA_VISIBLE_DEVICES=$i python render_splats.py  --splats_dir data/real_benchmark_frames_160_splats --out_dir data/real_benchmark_frames_160_rendered --process_id $i &
done

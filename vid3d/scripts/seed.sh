#!/bin/bash

# Define the starting and ending index
start=0
end=7

# Loop through the desired range of indices
for i in $(seq $start $end); do
    echo "Running seed_scene.py with process-id: $i"
    CUDA_VISIBLE_DEVICES=$i python seed_scene.py --seed-dir "data/real_benchmark_frames" --output-dir "data/real_benchmark_motion_160_seeds" --motion-id 160 --process-id $i &
done

#!/bin/bash

# Define the starting and ending index
start=0
end=7

# Loop through the desired range of indices
for i in $(seq $start $end); do
    echo "Running frame_to_multi_view.py with process-id: $i"
    CUDA_VISIBLE_DEVICES=$i python multi_view_to_splat.py  -w --sh_degree 0 --iterations 4000 --lambda_dssim 1.0 --lambda_lpips 0.0 --save_iterations 4000 --num_pts 100_000 --multi_view_dir data/multi_view --out_dir data/splats --model_path data/debug --sub_sample_frames 18 --process_id $i &
done

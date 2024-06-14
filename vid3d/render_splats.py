#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import torch
from gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from PIL import Image
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel

def render_set(out_dir, model_path, views, gaussians, pipeline, background):
    example_name = "/".join(model_path.split("/")[2:])
    render_path = os.path.join(out_dir, example_name)
    # render_path = os.path.join(model_path, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'view_{idx}' + ".png"))

def render_sets(out_dir, dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        images = []
        for _ in range(18):
            # Create a blank image with the specified color
            img = Image.new('RGB', (512, 512), (255, 255, 255))
            images.append(img)
        dataset.images = images
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(out_dir, dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    base_parser = ArgumentParser()
    base_parser.add_argument("--splats_dir", type=str, required=True)
    base_parser.add_argument("--out_dir", type=str, required=True)
    base_parser.add_argument("--process_id", type=int, required=True)
    base_args = base_parser.parse_args()

    model_dirs = set()
    for root, dirs, files in os.walk(base_args.splats_dir):
        for file in files:
            if file.endswith("cfg_args"):  # Check if the filename matches
                model_dirs.add(root)

    model_dirs = list(model_dirs) 
    model_dirs = sorted(model_dirs)
    chunk_size = len(model_dirs) // 8
    model_dirs = model_dirs[base_args.process_id * chunk_size:(base_args.process_id + 1) * chunk_size]

    for model_dir in tqdm(model_dirs):

        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--splats_dir", type=str, required=True)
        parser.add_argument("--out_dir", type=str, required=True)
        parser.add_argument("--process_id", type=int, required=True)

        sys.argv += ["--model_path", model_dir]

        args = get_combined_args(parser)
        # print("Rendering " + args.model_path)



        # import sys; sys.exit()

        # Initialize system state (RNG)
        safe_state(True)

        cur_lp = model.extract(args)
        cur_lp.num_frames = 10

        render_sets(args.out_dir, model.extract(args), args.iteration, pipeline.extract(args))

        sys.argv = sys.argv[:-2]

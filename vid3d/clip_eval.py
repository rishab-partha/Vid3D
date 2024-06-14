import os
import re
from typing import List
from argparse import ArgumentParser
from tqdm import tqdm
import pickle

import torch
import numpy as np
from torch import Tensor
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

def fetch_clipmodel(clip_model_name: str):
    model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    return model, processor

@torch.no_grad()
def clip_image_score(images: List[Tensor], model: CLIPModel, processor: CLIPImageProcessor):
    device = images[0].device
    processed_images = processor(images = images, return_tensors="pt")
    features = model.get_image_features(processed_images["pixel_values"].to(device))
    features /= features.norm(p=2, dim=-1, keepdim=True)
    score = 100*((features * features[0]).sum(axis = -1))

    score = score.mean(0)
    return torch.max(score, torch.zeros_like(score))

def multi_view_clipscore(directory: str, model: CLIPModel, processor: CLIPImageProcessor):
    reference_image = pil_to_tensor(Image.open(os.path.join(directory, "frame_0", "view_0.png"))).to("cuda")
    images = {i: [reference_image] for i in range(10)}
    for timestep in range(25):
        timestep_path = os.path.join(directory, f"frame_{timestep}")
        for view_idx in range(10):
            image = Image.open(os.path.join(timestep_path, f"view_{view_idx}.png"))
            images[view_idx].append(pil_to_tensor(image).to("cuda"))

    scores = []
    for view in range(10):
        scores.append(clip_image_score(images[view], model, processor).item())
    
    return scores


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--render-dir", type=str, required=True)
    parser.add_argument("--save-name", type=str, required=True)
    args = parser.parse_args()

    render_bases = set()
    for root, dirs, files in os.walk(args.render_dir):
        for file in files:
            if file.endswith(".png"):  # Check if the filename matches
                render_bases.add(root.split("/")[2])
    
    render_bases = list(render_bases) 
    render_bases = sorted(render_bases)

    model, processor = fetch_clipmodel("openai/clip-vit-base-patch32")

    model = model.to("cuda")

    all_scores = {}

    for render_base in tqdm(render_bases):
        render_dir = os.path.join(args.render_dir, render_base)

        scores = multi_view_clipscore(render_dir, model, processor)
        mean_score = np.mean(scores)

        all_scores[render_base] = {
            "all-views": scores,
            "mean": mean_score
        }

        print(f"{render_base}: {mean_score}")

    total_avg = 0
    for render_base, scores in all_scores.items():
        total_avg += scores["mean"]
        print(f"{render_base}: {scores['mean']}")
    
    print(f"Total average: {total_avg / len(all_scores)}")

    save_path = os.path.join("results", f"{args.save_name}.pkl")
    print(f"Saving scores to {save_path}")
    
    with open(save_path, "wb") as f:
        pickle.dump(all_scores, f)
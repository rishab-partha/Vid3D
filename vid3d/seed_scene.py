import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image

def main(args):

    # Load video model
    print("Loading video model...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    ).to(device="cuda")

    # Build list of seed frames
    seed_files = [file for file in os.listdir(args.seed_dir)]
    seed_files = sorted(seed_files)
    chunk_size = len(seed_files) // 8
    seed_files = seed_files[args.process_id * chunk_size:(args.process_id + 1) * chunk_size]

    # Process all video frames
    print("Generating videos...")

    for seed_file in tqdm(seed_files):
        image = load_image(f"{args.seed_dir}/{seed_file}/image.jpg")
        image = image.resize((512, 512))

        generator = torch.manual_seed(42)

        frames = pipe(image, decode_chunk_size=8, generator=generator, width=512, height=512, motion_bucket_id=args.motion_id).frames[0]

        seed_file_base = seed_file
        store_dir = f"{args.output_dir}/{seed_file_base}"
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        for i, frame in enumerate(frames):
            frame.save(f"{store_dir}/frame_{i}.png")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--seed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--motion-id", type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.process_id)

    main(args)

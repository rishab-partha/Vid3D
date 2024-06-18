# Vid3D: Synthesis of Dynamic 3D Scenes using 2D Video Diffusion
Rishab Parthasarathy<sup>1</sup>, Zachary Ankner<sup>1,2</sup>, Aaron Gokaslan<sup>2,3</sup>

<sup>1</sup>Massachusetts Institute of Technology, <sup>2</sup>Databricks Mosaic Research, <sup>3</sup>Cornell University

This repository contains the official implementation of [Vid3D: Synthesis of Dynamic 3D Scenes using 2D Video Diffusion](https://arxiv.org/abs/2406.). 

### [arXiv](https://arxiv.org/abs/2406.) | [Project Page](https://rishab-partha.github.io/Vid3D)

## Abstract

A recent frontier in computer vision has been the task of 3D video generation, which consists of generating a time-varying 3D representation of a scene. To generate dynamic 3D scenes, current methods explicitly model 3D temporal dynamics by jointly
optimizing for consistency across both time and
views of the scene. In this paper, we instead
investigate whether it is necessary to explicitly enforce multiview consistency over time, as current
approaches do, or if it is sufficient for a model to
generate 3D representations of each timestep independently. We hence propose a model, Vid3D, that
leverages 2D video diffusion to generate 3D videos
by first generating a 2D ”seed” of the video’s temporal dynamics and then independently generating
a 3D representation for each timestep in the seed
video. We evaluate Vid3D against two state-ofthe-art 3D video generation methods and find that
Vid3D is achieves comparable results despite not
explicitly modeling 3D temporal dynamics. We
further ablate how the quality of Vid3D depends on
the number of views generated per frame. While
we observe some degradation with fewer views,
performance degradation remains minor. Our
results thus suggest that 3D temporal knowledge
may not be necessary to generate high-quality
dynamic 3D scenes, potentially enabling simpler
generative algorithms for this task.


## Examples

https://github.com/rishab-partha/Vid3D/assets/56666587/7b0470c2-7a85-4c32-99c7-31185e239117

https://github.com/rishab-partha/Vid3D/assets/56666587/f271f623-6def-4140-bd50-5b7b2a281378

https://github.com/rishab-partha/Vid3D/assets/56666587/48d9d463-e0b7-4bf6-b753-d276f4e40295

https://github.com/rishab-partha/Vid3D/assets/56666587/47e5b4dd-bd90-4385-9e2a-8b5c9bfbd182

https://github.com/rishab-partha/Vid3D/assets/56666587/5515cb92-ad7f-48d2-9bcb-64d3552203a0

https://github.com/rishab-partha/Vid3D/assets/56666587/69895d68-d917-4e04-b7be-edd03798e581

https://github.com/rishab-partha/Vid3D/assets/56666587/ac8c9ef1-ef35-4d12-9ebb-3f884438f14f

https://github.com/rishab-partha/Vid3D/assets/56666587/4aa990de-edf5-4058-97b1-0c37dd16969e

## Instructions:
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download the weights for multi-view generation (from the great [V3D paper](https://github.com/heheyas/V3D))
```
wget https://huggingface.co/heheyas/V3D/resolve/main/V3D.ckpt -O ckpts/V3D_512.ckpt
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors -O ckpts/svd_xt.safetensors
```
3. We provide scripts to run our code on a node with 8 GPUs, which must have 80 GB of RAM. First, generate the seed videos.
```
cd vid3d
bash scripts/seed.sh
```
4. Generate multi-views of each frame.
```
bash scripts/frame_to_multi_view.sh
```
5. Convert multi-views to Gaussian Splats.
```
bash scripts/multi_view_to_splat.sh
```
6. Render the gaussian splats from varying angles.
```
bash scripts/render_splat.sh
```

## Citation:
If you found our work useful, please consider citing us at
```bibtex
@article{parthasarathy2024vid3d,
  author    = {Rishab Parthasarathy and Zachary Ankner and Aaron Gokaslan},
  title     = {Vid3D: Synthesis of Dynamic 3D Scenes using 2D Video Diffusion},
  journal   = {arXiv preprint arXiv:2406.},
  year      = {2024}
}
```

## Acknowledgements
We thank the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [V3D](https://github.com/heheyas/V3D) for the code bases that this project is based upon.

This project also began as a class project for the Advances in Computer Vision class at MIT, and we would like to thank Professors Sara Beery, Kaiming He, Mina Konakovic Lukovic, and Vincent Sitzmann for teaching the class, along with Joanna Materzynska and Emily Robinson for valuable feedback.

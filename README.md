# ARENA Practice Basics

Personal practice workspace for selected ARENA fundamentals exercises.

This repository currently contains local, notebook-based workspaces for ray tracing and CNNs/ResNets. The notebooks are kept close to the ARENA exercise structure, but trimmed down to setup code, problem statements, skeleton implementations, tests, and local assets.

## Contents

- `01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb` - ray tracing practice notebook.
- `01_ray_tracing/tests.py` - lightweight tests used by the notebook.
- `01_ray_tracing/utils.py` - plotting helpers for visualizing rays and geometry.
- `01_ray_tracing/pikachu.pt` and `01_ray_tracing/pikachu.stl` - mesh assets used for the final render.
- `02_cnns_resnets/0.2_CNNs_&_ResNets_exercises.ipynb` - CNNs and ResNets practice notebook.
- `02_cnns_resnets/tests.py` - local tests for module, convolution, pooling, and ResNet exercises.
- `02_cnns_resnets/utils.py` - display helpers used by the notebook.
- `02_cnns_resnets/imagenet_labels.json` and `02_cnns_resnets/resnet_inputs/` - assets for pretrained ResNet prediction exercises.

## What This Covers

- Building batches of 1D and 2D rays.
- Solving ray-line and ray-triangle intersections with `torch.linalg.solve`.
- Vectorizing intersection checks across many rays and scene objects.
- Rendering a triangle mesh by finding the nearest intersecting surface.
- Implementing basic PyTorch modules such as `ReLU`, `Linear`, `Conv2d`, `BatchNorm2d`, and pooling.
- Building and checking a ResNet-style architecture.
- Using pretrained ResNet features for prediction and feature extraction.
- Implementing low-level tensor operations for convolution and pooling.

## Setup

Create and activate a virtual environment, then install the small dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb
02_cnns_resnets/0.2_CNNs_&_ResNets_exercises.ipynb
```

## Notes

The notebooks are intentionally kept close to the ARENA exercise structure, with website-specific scaffolding removed so the focus stays on local practice.

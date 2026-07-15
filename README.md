# ARENA Practice Basics

This repository contains my local practice work for selected ARENA Chapter 0 fundamentals exercises. The focus is on implementing core ideas directly in PyTorch, checking them with tests, and keeping the notebooks runnable without the original course website scaffolding.

The two completed sections are:

- Ray tracing from first principles
- CNNs, ResNets, and basic neural network components

## What I Practiced

### Ray Tracing

The ray tracing notebook builds a small renderer using tensor operations. It starts with simple ray construction and ends with rendering a triangle mesh.

Main things demonstrated:

- Constructing batches of 1D and 2D camera rays.
- Solving ray-line and ray-triangle intersections with `torch.linalg.solve`.
- Vectorizing geometry computations across many rays and scene objects.
- Handling edge cases such as singular systems and non-forward intersections.
- Rendering a triangle mesh by selecting the nearest intersecting surface.

Notebook:

- `01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb`

Supporting files:

- `01_ray_tracing/tests.py`
- `01_ray_tracing/utils.py`
- `01_ray_tracing/pikachu.pt`
- `01_ray_tracing/pikachu.stl`

### CNNs and ResNets

The CNNs/ResNets notebook implements neural network building blocks and assembles them into larger models. The emphasis is on understanding layers as functions with stored parameters, then using those layers to build a ResNet-style architecture.

Main things demonstrated:

- Implementing PyTorch-style modules: `ReLU`, `Linear`, `Flatten`, `Sequential`, `Conv2d`, `BatchNorm2d`, and pooling.
- Training a simple MLP on MNIST with a validation loop.
- Understanding dataloaders, losses, optimizers, gradients, and training/evaluation modes.
- Implementing residual blocks with left/right branches and skip connections.
- Assembling `BlockGroup` and `ResNet34` from smaller components.
- Comparing a custom ResNet implementation against PyTorch's reference model.
- Running pretrained ResNet prediction checks using local ImageNet labels and sample images.

Notebook:

- `02_cnns_resnets/0.2_CNNs_&_ResNets_exercises.ipynb`

Supporting files:

- `02_cnns_resnets/tests.py`
- `02_cnns_resnets/utils.py`
- `02_cnns_resnets/plotly_utils.py`
- `02_cnns_resnets/imagenet_labels.json`
- `02_cnns_resnets/resnet_inputs/`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open one of:

```text
01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb
02_cnns_resnets/0.2_CNNs_&_ResNets_exercises.ipynb
```

## Notes

- The notebooks are adapted from ARENA fundamentals exercises, but trimmed for local practice.
- Downloaded datasets such as MNIST are ignored via `.gitignore`.
- The local notebooks use the `ARENA (.venv)` Jupyter kernel.

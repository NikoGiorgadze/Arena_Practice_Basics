# ARENA Practice Basics

Personal solutions and notes for selected ARENA fundamentals exercises.

This repository currently focuses on the ray tracing practice section. The notebook builds a small ray tracer from first principles using PyTorch tensors, starting with simple ray construction and ending with triangle mesh rendering.

## Contents

- `01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb` - worked ray tracing notebook.
- `01_ray_tracing/tests.py` - lightweight tests used by the notebook.
- `01_ray_tracing/utils.py` - plotting helpers for visualizing rays and geometry.
- `01_ray_tracing/pikachu.pt` and `01_ray_tracing/pikachu.stl` - mesh assets used for the final render.

## What This Covers

- Building batches of 1D and 2D rays.
- Solving ray-line and ray-triangle intersections with `torch.linalg.solve`.
- Vectorizing intersection checks across many rays and scene objects.
- Rendering a triangle mesh by finding the nearest intersecting surface.

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
```

## Notes

The notebook is intentionally kept close to the ARENA exercise structure, with extra comments added while working through the geometry and tensor-shape reasoning.

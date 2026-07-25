# ARENA Chapter 0 Practice

This repository contains my local practice work for selected ARENA Chapter 0 fundamentals exercises. The notebooks focus on implementing the underlying ideas directly, checking them with tests, and staying runnable without the original course website scaffolding.

## Repository Map

| Section | Main topics | Notebook |
| --- | --- | --- |
| Ray tracing | Batched rays, intersections, triangle meshes, rendering | [`01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb`](01_ray_tracing/0.1_Ray_Tracing_exercises.ipynb) |
| CNNs and ResNets | PyTorch-style modules, convolutions, residual blocks, ResNet34 | [`02_cnns_resnets/0.2_CNNs_&_ResNets_exercises.ipynb`](02_cnns_resnets/0.2_CNNs_%26_ResNets_exercises.ipynb) |
| Optimisation | SGD-family optimisers, feature extraction, W&B tracking and sweeps | [`03_optimization/0.3_Optimization_exercises.ipynb`](03_optimization/0.3_Optimization_exercises.ipynb) |

## What I Practiced

### Ray Tracing

- Constructing batches of 1D and 2D camera rays.
- Solving ray-line and ray-triangle intersections with `torch.linalg.solve`.
- Vectorizing geometry calculations across many rays and scene objects.
- Handling singular systems and non-forward intersections.
- Rendering a triangle mesh by selecting the nearest intersecting surface.

### CNNs and ResNets

- Implementing PyTorch-style `ReLU`, `Linear`, `Flatten`, `Sequential`, `Conv2d`, `BatchNorm2d`, and pooling modules.
- Training an MLP on MNIST with separate training and validation loops.
- Building residual blocks, block groups, and a complete ResNet34.
- Comparing the custom architecture and pretrained parameters with torchvision.
- Reusing the completed model from [`resnet_model.py`](02_cnns_resnets/resnet_model.py) for later feature-extraction exercises.

### Optimisation

- Applying gradient descent to two-dimensional loss landscapes.
- Implementing SGD with momentum and weight decay, RMSprop, Adam, and AdamW.
- Comparing optimiser behavior and hyperparameters.
- Fine-tuning the final classification layer of a pretrained ResNet34 on CIFAR-10.
- Recording experiments locally or with Weights & Biases, including configurable sweeps.
- Separating expensive training and sweep launches from class definitions so runs remain optional.

#### The Two ResNet Trainers

`ResNetFinetuner` is the base trainer. It contains the actual training machinery: model and dataset setup, optimizer construction, minibatch training, evaluation, and local loss and accuracy records.

```python
class WandbResNetFinetuner(ResNetFinetuner):
```

`WandbResNetFinetuner` inherits the base trainer's methods and attributes unless it overrides them. It uses the same model, forward pass, cross-entropy loss, backpropagation, optimizer update, and evaluation procedure, while adding Weights & Biases experiment management.

Its setup begins with:

```python
super().pre_training_setup()
```

This runs the parent setup first—constructing the model, optimizer, datasets, and data loaders—before the subclass starts or connects to a W&B run and calls `wandb.watch`.

The argument classes follow the same pattern. `ResNetFinetuningArgs` holds ordinary hyperparameters such as batch size, epochs, learning rate, and weight decay. `WandbResNetFinetuningArgs` inherits those fields and adds the W&B project and run names.

| | `ResNetFinetuner` | `WandbResNetFinetuner` |
| --- | --- | --- |
| Training computation | Standard forward, backward, and update steps | The same computation |
| Metric storage | Local lists in `logged_variables` | `wandb.log` |
| Visualization | Local plotting after a run | Persistent W&B dashboards |
| Experiment comparison | Manual | W&B runs and sweeps |
| Extra dependency | None beyond the training stack | `wandb` and a W&B account |
| Best suited for | Quick local experiments | Tracked and comparable experiments |

In short:

```text
ResNetFinetuner      = ordinary training with local experiment records
WandbResNetFinetuner = the same training procedure plus W&B experiment management
```

The W&B trainer uses `wandb.init`, can monitor parameters and gradients with `wandb.watch`, records metrics with `wandb.log`, and closes the run with `wandb.finish`. W&B does not itself improve model accuracy; it improves experiment organization, visualization, persistence, and comparison.

#### Example W&B Run

The optional random-sweep workflow was verified with one completed trial:

| Run | Batch size | Learning rate | Weight decay | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| [`fancy-sweep-1`](https://wandb.ai/niko-bnt/day3-resnet-sweep/runs/lh80a4ia) | 128 | 0.00219 | 0.00437 | 80.83% |

This is a single sampled feature-extraction run rather than a benchmark. Its dashboard preserves the loss and accuracy history together with parameter and gradient monitoring. Raw `wandb/` files remain local and are ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

The notebooks automatically prefer Apple MPS, then CUDA, then CPU when accelerator selection is needed.

Weights & Biases is optional. To use its experiment-tracking cells:

```bash
wandb login
```

Downloaded datasets and local experiment files are deliberately excluded from Git. The optimisation notebook expects CIFAR-10 under `data/cifar-10-batches-py`; on a fresh checkout, download or place the dataset there before running the ResNet training cells.

## Notes

- These notebooks are adapted and trimmed for local practice.
- Supporting `tests.py` and utility files stay beside the relevant notebook.
- Long-running training and W&B launches are kept in explicit optional cells.
- The local notebooks use the `ARENA (.venv)` Jupyter kernel.

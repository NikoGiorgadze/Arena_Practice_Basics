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

#### Optimisation as Dynamics in Parameter Space

Collect every trainable scalar in a model into one vector

$$
\theta \in \mathbb{R}^d.
$$

Training constructs a discrete trajectory $\theta_0,\theta_1,\theta_2,\ldots$ through this parameter space, with the aim of reducing a scalar loss $L(\theta)$. It is often useful to introduce an artificial **optimisation time** $s$ and imagine a continuous curve $s\mapsto\theta(s)$. This is an algorithmic parameter, not physical time.

The simplest continuous model is gradient flow:

$$
\frac{d\theta}{ds}=-\nabla L(\theta).
$$

It is an overdamped dissipative system because

$$
\frac{dL}{ds}
=
\nabla L(\theta)^T\frac{d\theta}{ds}
=
-\lVert\nabla L(\theta)\rVert^2
\leq 0.
$$

Gradient descent is its forward-Euler discretisation:

$$
\theta_t=\theta_{t-1}-\eta\nabla L(\theta_{t-1}),
\qquad t=1,2,\ldots,
$$

so the learning rate $\eta$ acts like a numerical integration step size. If the Hessian $H=\nabla^2L$ has very different eigenvalues, parameter space contains stiff high-curvature directions and soft low-curvature directions. A step large enough to move efficiently along a soft direction can oscillate or diverge along a stiff one; for a positive quadratic, stability requires $\eta<2/\lambda_{\max}(H)$. This is why an ill-conditioned landscape is difficult to navigate with one global learning rate.

For minibatch training, take $L$ to mean the full-data or population objective. The optimizer receives a stochastic estimate

$$
g_t
=
\nabla L(\theta_{t-1})+\xi_t,
$$

where update calls are numbered $t=1,2,\ldots$ and $\xi_t\in\mathbb{R}^d$ is the gradient-noise vector caused by sampling a minibatch. The minibatch itself is a collection of examples or indices, not a vector in parameter space; its gradient is. If the minibatch estimator is unbiased for $L$, then $\mathbb{E}[\xi_t\mid\theta_{t-1}]=0$, but its covariance is generally anisotropic and depends on both the data and the current parameters. SGD can therefore be viewed approximately as a noisy or Langevin-like discretisation, but minibatch noise is not generally isotropic equilibrium thermal noise. In the notebook's two-dimensional loss-landscape examples the gradient is exact, so the class named `SGD` behaves as deterministic gradient descent; stochasticity appears only when a `DataLoader` supplies sampled minibatches.

#### Update Rules Implemented in the Notebook

Below, step $t$ receives $\theta_{t-1}$ and produces $\theta_t$, while $g_t$ denotes the gradient stored in `param.grad` at the start of that step. All vector squares, square roots, and divisions are elementwise. The symbols $\eta$, $\mu$, $\lambda$, $\alpha$, $\epsilon$, and $(\beta_1,\beta_2)$ correspond respectively to the constructor arguments `lr`, `momentum`, `weight_decay`, `alpha`, `eps`, and `betas`. Buffers satisfy $b_0=m_0=v_0=0$. For SGD, RMSprop, and Adam, the implementation first forms the **coupled** weight-decayed gradient

$$
\tilde g_t=g_t+\lambda\theta_{t-1}.
$$

AdamW instead uses the raw gradient in its moment estimates and applies decay directly to the parameter.

| Optimizer | Persistent state | Weight-decay convention | Epsilon placement |
| --- | --- | --- | --- |
| SGD | momentum buffer $b_t$ | coupled through $\tilde g_t$ | — |
| RMSprop | squared-gradient buffer $v_t$ and optional momentum buffer $b_t$ | coupled through $\tilde g_t$ | $\sqrt{v_t}+\epsilon$ |
| Adam | first moment $m_t$, second moment $v_t$, and step $t$ | coupled through $\tilde g_t$ | $\sqrt{\hat v_t}+\epsilon$ |
| AdamW | first moment $m_t$, second moment $v_t$, and step $t$ | decoupled multiplicative decay | $\sqrt{\hat v_t}+\epsilon$ |

**SGD and momentum.** The repository uses the same basic momentum-buffer convention as PyTorch SGD with zero dampening and without Nesterov momentum:

$$
b_t=\mu b_{t-1}+\tilde g_t,
\qquad
\theta_t=\theta_{t-1}-\eta b_t.
$$

When $\mu=0$, the update uses $\tilde g_t$ directly. Momentum is related to the damped heavy-ball system

$$
M\ddot\theta+\Gamma\dot\theta+\nabla L(\theta)=0,
$$

where the mass or inertia matrix $M$ is symmetric positive definite and the friction matrix $\Gamma$ is symmetric positive semidefinite. This follows from the mechanical Lagrangian and Rayleigh dissipation function

$$
\mathcal L_{\mathrm{mech}}
=
\frac12\dot\theta^TM\dot\theta-L(\theta),
\qquad
\mathcal R
=
\frac12\dot\theta^T\Gamma\dot\theta.
$$

The analogy explains why momentum helps in narrow valleys: rapidly alternating transverse gradient components cancel in $b_t$, while a consistently directed gradient component accumulates and produces a sustained downhill update through $-\eta b_t$. The stored PyTorch-style buffer has gradient units; it is not literally the mechanical velocity. The parameter displacement is $-\eta b_t$, so a velocity-like discrete variable differs from $b_t$ by a factor involving the learning rate.

**RMSprop.** After coupled decay, this implementation updates

$$
v_t
=
\alpha v_{t-1}+(1-\alpha)\tilde g_t^2,
\qquad
q_t
=
\frac{\tilde g_t}{\sqrt{v_t}+\epsilon}.
$$

If momentum is enabled, it then applies

$$
b_t=\mu b_{t-1}+q_t,
\qquad
\theta_t=\theta_{t-1}-\eta b_t;
$$

otherwise $\theta_t=\theta_{t-1}-\eta q_t$. The squared-gradient buffer is an exponentially weighted **uncentred second moment**, not necessarily a variance:

$$
v_t
=
(1-\alpha)\sum_{j=0}^{t-1}\alpha^j\tilde g_{t-j}^2.
$$

The age weights $p_j=(1-\alpha)\alpha^j$ decay geometrically, so recent gradients matter most. Over the finite history they sum to $1-\alpha^t$; the missing $\alpha^t$ is the weight of the zero initial state. RMSprop acts as a history-dependent diagonal preconditioner or mobility matrix,

$$
D_t
=
\operatorname{diag}
\left(
\frac{1}{\sqrt{v_{t,i}}+\epsilon}
\right),
$$

which reduces steps in coordinates that have recently had large gradients. A continuous relaxation is

$$
\tau_v\dot v=\tilde g^2-v,
$$

where $v$ is an auxiliary dynamical variable with finite response time. If $\tilde g^2$ is treated as constant over one interval, exact integration gives $\alpha=e^{-\Delta s/\tau_v}$; a forward-Euler discretisation instead gives $\alpha=1-\Delta s/\tau_v$. RMSprop is not Newton's method and does not estimate the Hessian directly.

**Adam.** Adam combines filtered first and second moments:

$$
m_t
=
\beta_1m_{t-1}+(1-\beta_1)\tilde g_t,
\qquad
v_t
=
\beta_2v_{t-1}+(1-\beta_2)\tilde g_t^2.
$$

Because both buffers begin at zero, their early values are biased toward zero. For an approximately stationary gradient mean, for example,

$$
\mathbb E[m_t]=(1-\beta_1^t)\mathbb E[\tilde g],
$$

which motivates the implemented corrections

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

The parameter update is

$$
\theta_t
=
\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Here $m_t$ is a low-pass-filtered gradient estimate—the negative of a mechanical force estimate—while $v_t$ tracks squared-gradient scale. Their continuous relaxations are

$$
\tau_m\dot m=\tilde g-m,
\qquad
\tau_v\dot v=\tilde g^2-v.
$$

Although $m_t$ is commonly called Adam's momentum, it is a gradient moving average rather than the mechanical momentum variable in heavy-ball dynamics.

**AdamW and decoupled decay.** For plain SGD without momentum, adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss gives

$$
\theta_t
=
(1-\eta\lambda)\theta_{t-1}-\eta g_t,
$$

so $L_2$ regularisation and direct multiplicative weight decay are equivalent. The equivalence fails once $\lambda\theta$ is passed through adaptive moment estimates (and is also altered by a momentum buffer). AdamW keeps regularisation out of the moments. In the exact order used here,

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

is applied first, then the ordinary bias-corrected Adam update computed from the raw gradient $g_t$ is subtracted. Decoupling makes the shrinkage independent of the coordinate-wise adaptive denominator.

#### From Equations to PyTorch

In the pseudocode below, an assignment arrow $x\leftarrow F(x)$ means: evaluate the right-hand side using the current state, then replace the stored value on the left.

```text
SGD
    g ← param.grad + λ param
    if μ ≠ 0: b ← μ b + g; g ← b
    param ← param - η g

RMSprop
    g ← param.grad + λ param
    v ← α v + (1 - α) g²
    g ← g / (sqrt(v) + ε)
    if μ > 0: b ← μ b + g; g ← b
    param ← param - η g

Adam
    g ← param.grad + λ param
    m ← β₁ m + (1 - β₁) g
    v ← β₂ v + (1 - β₂) g²
    m_hat ← m / (1 - β₁ᵗ)
    v_hat ← v / (1 - β₂ᵗ)
    param ← param - η m_hat / (sqrt(v_hat) + ε)
    after all parameters: t ← t + 1

AdamW
    g ← param.grad
    param ← (1 - ηλ) param
    m ← β₁ m + (1 - β₁) g
    v ← β₂ v + (1 - β₂) g²
    m_hat ← m / (1 - β₁ᵗ)
    v_hat ← v / (1 - β₂ᵗ)
    param ← param - η m_hat / (sqrt(v_hat) + ε)
    after all parameters: t ← t + 1
```

For both Adam classes, `t` starts at one, is shared by all parameters, and is incremented once after the optimizer has processed the full parameter list. The implementations also assume that every listed parameter has a non-`None` gradient when `step()` is called.

The corresponding PyTorch objects and operations are:

| Mathematical object | PyTorch representation |
| --- | --- |
| parameter $\theta$ | `torch.nn.Parameter` |
| gradient $\nabla_\theta L$ | `param.grad` |
| reverse-mode or adjoint differentiation | `loss.backward()` |
| discrete integration/update step | `optimizer.step()` |
| auxiliary dynamical variables | momentum and moment buffers |
| reset derivative storage | `optimizer.zero_grad()` |

Gradients accumulate by default: repeated calls to `backward()` add into `param.grad`. The notebook's `zero_grad()` sets each gradient to `None`, so the next backward pass creates fresh gradient storage.

An optimizer is stateful. It converts the supplied parameter iterable to a list of references and retains persistent tensors such as $b_t$, $m_t$, and $v_t$ between calls to `step()`. Updating a registered parameter therefore requires in-place mutation. Python rebinding,

```python
p = p - lr * grad
```

makes `p` refer to a new tensor, while

```python
p.add_(grad, alpha=-lr)
```

mutates the parameter object already registered in the model. PyTorch marks in-place methods with a trailing underscore, as in `add_`, `mul_`, `copy_`, and `zero_`. The notebook uses operations such as `copy_` for buffers and `theta -= ...` for parameters.

Optimizer updates run under `torch.inference_mode()` so the mutations do not become part of the autograd graph and do not trigger forbidden gradient-tracked in-place operations on leaf parameters. Tensor aliasing also matters: after `g = p.grad`, both names refer to the same tensor, so changing `g` in place would change the stored gradient. This is why coupled decay is formed out of place as `g = g + weight_decay * p`.

Finally, a trajectory needs independent snapshots. Appending the live parameter tensor repeatedly would store aliases to one object whose value keeps changing, while

```python
trajectory.append(tensor.detach().clone())
```

disconnects the snapshot from autograd and gives it independent storage.

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

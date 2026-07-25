"""Reusable model definitions extracted from the completed CNNs & ResNets notebook."""

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torchvision import models


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0.0))


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        weight = (2 * t.rand(out_features, in_features) - 1) / np.sqrt(in_features)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = (2 * t.rand(out_features) - 1) / np.sqrt(out_features)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = einops.einsum(x, self.weight, "... i, k i -> ... k")
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            scale * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join(f"{key}={getattr(self, key)}" for key in keys)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["kernel_size", "stride", "padding"]
        return ", ".join(f"{key}={getattr(self, key)}" for key in keys)


class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, module in enumerate(modules):
            self._modules[str(index)] = module

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class BatchNorm2d(nn.Module):
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        mean = einops.rearrange(mean, "c -> 1 c 1 1")
        var = einops.rearrange(var, "c -> 1 c 1 1")
        weight = einops.rearrange(self.weight, "c -> 1 c 1 1")
        bias = einops.rearrange(self.bias, "c -> 1 c 1 1")
        x_hat = (x - mean) / t.sqrt(var + self.eps)
        return weight * x_hat + bias

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"num_features={self.num_features}",
                f"eps={self.eps}",
                f"momentum={self.momentum}",
            ]
        )


class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.mean(x, dim=(2, 3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        is_shape_preserving = first_stride == 1 and in_feats == out_feats

        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats),
        )
        self.right = (
            nn.Identity()
            if is_shape_preserving
            else Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats),
            )
        )
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.left(x) + self.right(x))


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        self.blocks = Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *[ResidualBlock(out_feats, out_feats) for _ in range(n_blocks - 1)],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.in_layers = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.residual_layers = Sequential(
            *[
                BlockGroup(
                    n_blocks=n_blocks_per_group[index],
                    in_feats=[64, *out_features_per_group][index],
                    out_feats=out_features_per_group[index],
                    first_stride=first_strides_per_group[index],
                )
                for index in range(len(n_blocks_per_group))
            ]
        )
        self.out_layers = Sequential(
            AveragePool(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layers(x)
        x = self.residual_layers(x)
        return self.out_layers(x)


def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.ResNet) -> ResNet34:
    """Copy torchvision ResNet34 parameters and buffers into the matching custom architecture."""
    custom_state = my_resnet.state_dict()
    pretrained_state = pretrained_resnet.state_dict()
    assert len(custom_state) == len(pretrained_state), "Mismatching state dictionaries."

    state_dict_to_load = {
        custom_key: pretrained_value
        for (custom_key, _), (_, pretrained_value) in zip(
            custom_state.items(), pretrained_state.items()
        )
    }
    my_resnet.load_state_dict(state_dict_to_load)
    return my_resnet


def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    """Return the custom pretrained ResNet34 with only a new classifier left trainable."""
    model = ResNet34()
    pretrained = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model = copy_weights(model, pretrained)
    model.requires_grad_(False)
    model.out_layers[-1] = Linear(model.out_features_per_group[-1], n_classes)
    return model

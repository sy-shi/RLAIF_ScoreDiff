import torch
import torch.nn as nn

from typing import Union, Tuple, Any

from ray.rllib.models.utils import get_activation_fn

# A simple convolutional 2D layer. Taken from RLlib: https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/models/torch/misc.py
class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        # Defaulting these to nn.[..] will break soft torch import.
        initializer: Any = "default",
        activation_fn: Any = "default",
        bias_init: float = 0,
    ):
        """Creates a standard Conv2d layer, similar to torch.nn.Conv2d
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: If int, the kernel is
                a tuple(x,x). Elsewise, the tuple can be specified
            stride: Controls the stride
                for the cross-correlation. If int, the stride is a
                tuple(x,x). Elsewise, the tuple can be specified
            padding: Controls the amount
                of implicit zero-paddings during the conv operation
            initializer: Initializer function for kernel weights
            activation_fn: Activation function at the end of layer
            bias_init: Initalize bias weights to bias_init const
        """
        super(SlimConv2d, self).__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)

        layers.append(nn.ReLU())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


# def create_slim_conv(config):
#     layers = []
#     # Padding layer.
#     if padding:
#         layers.append(nn.ZeroPad2d(padding))
#     # Actual Conv2D layer (including correct initialization logic).
#     conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
#     if initializer:
#         if initializer == "default":
#             initializer = nn.init.xavier_uniform_
#         initializer(conv.weight)
#     nn.init.constant_(conv.bias, bias_init)
#     layers.append(conv)

#     layers.append(nn.ReLU())
#     # Put everything in sequence.
#     self._model = nn.Sequential(*layers)
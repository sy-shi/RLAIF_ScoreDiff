import model.resnet
import torch
import torch.nn as nn
import torchvision

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, SlimConv2d
from ray.rllib.utils.torch_utils import FLOAT_MIN




class ActorCritic(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.action_masking = model_config["custom_model_config"]["action_masking"]
        self.critic_share_layers = model_config["custom_model_config"]["critic_share_layers"]
        self.input_embedding_size = model_config["custom_model_config"]["actor_layer_sizes"][0][0]

        self.input_layers = self._create_convolutional_layers(
            model_config["custom_model_config"]["input_conv_channels"],
            model_config["custom_model_config"]["conv_filters"],
            self.input_embedding_size
        )

        self.actor_layers = self._create_dense_layers(
            model_config["custom_model_config"]["actor_layer_sizes"],
            activation_at_end=False
        )

        self.critic_layers = self._create_dense_layers(
            model_config["custom_model_config"]["critic_layer_sizes"],
            activation_at_end=False
        )

        if not self.critic_share_layers:
            self.critic_input_layers = self._create_convolutional_layers(
                model_config["custom_model_config"]["input_conv_channels"],
                model_config["custom_model_config"]["conv_filters"],
                self.input_embedding_size
            )
        else:
            self.critic_input_layers = self.input_layers

        self._features = None

    def _create_convolutional_layers(self, in_channel, conv_filters, embedding_size):
        if isinstance(conv_filters, list):
            layers = []

            prev_out = in_channel

            for out_channel, kernel, stride in conv_filters:
                if not layers:
                    padding = (1, 2, 1, 2)
                else:
                    padding = None

                if out_channel == "pool":
                    layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride),)
                else:
                    layers.append(SlimConv2d(prev_out, out_channel, kernel, stride, padding))
                    prev_out = out_channel

            layers = nn.ModuleList(layers)
        elif "resnet" in conv_filters:
            layers = model.resnet.create_convolutional_layers(conv_filters, in_channel, embedding_size)
        else:
            raise("Unknown conv_filter type.")

        return layers

    def _create_dense_layers(self, sizes, layer_type = nn.Linear, activation_type = nn.ReLU, initializer = normc_initializer, activation_at_end=True):
        layers = []

        for idx, (in_size, out_size) in enumerate(sizes):
            layers.append(layer_type(in_size, out_size))

            if initializer is not None:
                initializer(layers[-1].weight)

            if activation_type is not None and (activation_at_end or idx < len(sizes)-1):
                layers.append(activation_type())

        layers = nn.ModuleList(layers)

        return layers

    def _compute_layers(self, x, layers):
        if isinstance(layers, nn.ModuleList):
            for layer in layers:
                x = layer(x)
        else:
            x = layers(x)
        
        return x

    def _preprocess_obs(self, input_dict):
        x = input_dict["obs"]["image"].float()

        # Reorder the state from (NHWC) to (NCHW).
        # Most images put channel last, but pytorch conv expects channel first.
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, input_dict, state, seq_lens):
        self._features = self._preprocess_obs(input_dict)

        x = self._compute_layers(self._features, self.input_layers)
        x = x.reshape(x.shape[0], -1)
        # print("**** New size: " + str(x.shape))
        x = self._compute_layers(x, self.actor_layers)

        if self.action_masking and "action_mask" in input_dict["obs"]:
            action_mask = input_dict["obs"]["action_mask"]

            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            x = x + inf_mask

        # Make sure to return logits. The output gets placed into a torch categorical distribution as logits (not probs). So do not softmax!
        return x, state

    def value_function(self, input_dict = None):
        if input_dict is not None:
            x = self._preprocess_obs(input_dict)
        else:
            x = self._features

        x = self._compute_layers(x, self.critic_input_layers)
        x = x.reshape(x.shape[0], -1)
        x = self._compute_layers(x, self.critic_layers).squeeze(1)

        return x

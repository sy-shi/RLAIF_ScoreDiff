import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch
import os

class RewardModel(nn.Module):
    def __init__(self, config):
        super(RewardModel, self).__init__()
        self.conv_layers = self._create_convolutional_layers(
            3,
            config["conv_filters"],
            config["conv_activation"]
        )
        # print(self.conv_layers)
        self.fc_size = self._get_conv_output()
        # print("fc_size: ", self.fc_size)

        self.fc_layers = self._create_dense_layers(
            config["fc_layer_sizes"],
            activation_at_end = True
        )
        self.clip_at_last = config["clip_at_last"]
        self.clip_scale = config["clip_scale"]
    
    def _get_conv_output(self):
        input = torch.rand([16,3,13,9])
        if self.conv_layers is not None:
            output = self._compute_layers(input, self.conv_layers)
            n_size = output.reshape(output.shape[0],-1).shape[1]
        else:
            n_size = 351
        return n_size


    def _create_convolutional_layers(self, in_channel, conv_filters, conv_activation):
        if len(conv_filters) == 0:
            return None
        layers = []

        prev_out = in_channel

        for out_channel, kernel, stride, padding in conv_filters:
            # if not layers:
            #     padding = (2, 1)
            # else:
            #     padding = (0,0)

            if out_channel == "pool":
                layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride),)
            else:
                layers.append(nn.Conv2d(prev_out, out_channel, kernel, stride, padding))
                prev_out = out_channel
                if conv_activation:
                    layers.append(nn.ReLU())

        layers = nn.ModuleList(layers)
        return layers

    def _create_dense_layers(self, sizes, layer_type = nn.Linear, activation_type = nn.ReLU, initializer = xavier_normal_, activation_at_end=True):
        layers = []

        for idx, (in_size, out_size) in enumerate(sizes):
            if idx == 0:
                layers.append(layer_type(self.fc_size, out_size))
            else:
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
    
    def forward(self, x):
        if self.conv_layers is not None:
            x = self._compute_layers(x, self.conv_layers)
        x = x.reshape(x.shape[0], -1)
        # print("**** New size: " + str(x.shape))
        x = self._compute_layers(x, self.fc_layers)
        if self.clip_at_last == "tanh":
            x = self.clip_scale * nn.functional.tanh(x)
        return x
    

class RandomNetwork(RewardModel):
    def __init__(self, config, file_path):
        assert file_path is not None
        super().__init__(config)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=next(self.parameters()).device))
        else:
            torch.save(self.state_dict(), file_path)
        
    def forward(self, x):
        x = super().forward(x)
        x = nn.functional.sigmoid(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, D=8, W=64, input_ch=3, output_ch=4, skips=[], final_activation=None, weight_norm=False):
        """

        :param G: number of parallel networks
        :param D: depth of a network
        :param W: width of a network
        :param input_ch: dimension of point embedding
        :param output_ch: dimension of output features
        :param skips: layers where to add a skip connection
        :param final_activation: nn.Module
        """
        super(MLP, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        ilayers = [input_ch, W] + [W] * (D-2)
        olayers = [W] + [W] * (D-1)
        for i in skips:
            ilayers[i+1] += input_ch

        if weight_norm:
            self.pts_linears = nn.ModuleList([
                nn.utils.weight_norm(nn.Linear(iW, oW), dim=0) for (iW, oW) in zip(ilayers, olayers)])
            self.out_layer = nn.utils.weight_norm(nn.Linear(olayers[-1], self.output_ch), dim=0)
        else:
            self.pts_linears = nn.ModuleList([nn.Linear(iW, oW) for (iW, oW) in zip(ilayers, olayers)])
            self.out_layer = nn.Linear(olayers[-1], self.output_ch)

        if final_activation is None:
            self.final_activation = nn.Identity()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        else:
            raise NotImplementedError('Please supply an nn.Module as the final activation layer!')

    def init_last_layer_with_small_value(self):
        """ initializes with small value for better starting
        """
        nn.init.normal_(self.out_layer.weight, mean=0., std=1e-4)

    def forward(self, x):
        """
        :param x: [batch_size, num_of_parallel_networks, num_of_points, channels]
        :param params
        :return: [batch_size, num_of_parallel_networks, num_of_points, 4 (rgb+density)]
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # h = F.softplus(h)
            if i in self.skips:
                h = torch.cat([h, x], dim=-1)

        output = self.out_layer(h)
        output = self.final_activation(output)
        return output

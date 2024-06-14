import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .parallel_linear import ParallelLinear, parallel_concat


class MetaMultiParallelMLP(nn.Module):
    def __init__(self, G=16, D=8, W=64, input_ch=3, output_ch=4, skips=[4], final_activation=None, weight_norm=False,
                 softplus=False):
        """

        :param G: number of parallel networks
        :param D: depth of a network
        :param W: width of a network
        :param input_ch: dimension of point embedding
        :param output_ch: dimension of output features
        :param skips: layers where to add a skip connection
        :param final_activation: nn.Module
        """
        super(MetaMultiParallelMLP, self).__init__()

        self.G = G
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        self.softplus = softplus

        ilayers = [input_ch, W] + [W] * (D-2)
        olayers = [W] + [W] * (D-1)
        for i in skips:
            ilayers[i+1] += input_ch

        if weight_norm:
            self.pts_linears = nn.ModuleList([
                nn.utils.weight_norm(ParallelLinear(G, iW, oW), dim=0) for (iW, oW) in zip(ilayers, olayers)])
            self.out_layer = nn.utils.weight_norm(ParallelLinear(G, olayers[-1], self.output_ch), dim=0)
        else:
            self.pts_linears = nn.ModuleList([ParallelLinear(G, iW, oW) for (iW, oW) in zip(ilayers, olayers)])
            self.out_layer = ParallelLinear(G, olayers[-1], self.output_ch)

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
        assert len(x.shape) == 4, 'input tensor should have a shape of [B, G, N, C] '
        assert x.shape[1] == self.G, \
            'input tensor should have %d parallel groups, but it has %s batches' % (self.G, x.shape[1])

        batch_size = x.shape[0]
        pt_num = x.shape[2]
        x = x.permute(0, 1, 3, 2).reshape(batch_size, self.G * self.input_ch, pt_num)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # h = F.softplus(h, beta=100) if self.softplus else F.relu(h)
            h = F.elu(h) if self.softplus else F.relu(h)
            # h = F.softplus(h)
            if i in self.skips:
                h = parallel_concat([x, h], self.G)

        output = self.out_layer(h)
        output = self.final_activation(output)
        output = output.view([batch_size, self.G, self.output_ch, pt_num]).permute(0, 1, 3, 2)   # [B, G, N, C]
        h = h.view([batch_size, self.G, -1, pt_num]).permute(0, 1, 3, 2)   # [B, G, N, C]
        return output, h


class MultiParallelNerfNet(nn.Module):
    def __init__(self, G=16, D=8, W=64, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False, softplus=False, weight_norm=False):
        """

        :param G: number of parallel networks
        :param D: depth of a network
        :param W: width of a network
        :param input_ch: dimension of point embedding
        :param input_ch_views: dimension of view embedding
        :param output_ch: dimension of output features
        :param skips: layers where to add a skip connection
        :param use_viewdirs: whether input view direction or not
        :param output_rgb_ch:
        :param output_alpha_ch:
        """
        super(MultiParallelNerfNet, self).__init__()

        self.G = G
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.softplus = softplus
        self.weight_norm = weight_norm

        ilayers = [input_ch, W] + [W] * (D-2)
        olayers = [W] + [W] * (D-1)
        for i in skips:
            ilayers[i+1] += input_ch

        parallel_linear = lambda G, iW, oW: nn.utils.weight_norm(ParallelLinear(G, iW, oW)) if self.weight_norm else ParallelLinear(G, iW, oW)

        self.pts_linears = nn.ModuleList([parallel_linear(G, iW, oW) for (iW, oW) in zip(ilayers, olayers)])

        if use_viewdirs:
            self.views_linears = nn.ModuleList([parallel_linear(G, input_ch_views + W, W // 2)])

            self.feature_linear = parallel_linear(G, W, W)
            self.rgb_linear = parallel_linear(G, W // 2, output_ch)
        else:
            self.output_linear = parallel_linear(G, W, output_ch)

    def forward(self, x):
        """
        :param x: [batch_size, num_of_parallel_networks, num_of_points, channels]
        :return: [batch_size, num_of_parallel_networks, num_of_points, 4 (rgb+density)]
        """
        assert len(x.shape) == 4, 'input tensor should have a shape of [B, G, N, C] '
        assert x.shape[1] == self.G, \
            'input tensor should have %d parallel groups, but it has %s batches' % (self.G, x.shape[1])

        batch_size = x.shape[0]
        pt_num = x.shape[2]
        if self.input_ch_views > 0:
            input_pts, input_views = \
                torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        input_pts = input_pts.permute(0, 1, 3, 2).reshape(batch_size, self.G * self.input_ch, pt_num)
        h = input_pts  # [B, G*C, N]
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # h = F.softplus(h, beta=10) if self.softplus else F.relu(h)
            h = F.elu(h) if self.softplus else F.relu(h)
            # h = F.tanh(h) if self.softplus else F.relu(h)
            if i in self.skips:
                h = parallel_concat([input_pts, h], self.G)

        if self.use_viewdirs:
            feature = self.feature_linear(h)
            input_views = input_views.permute(0, 1, 3, 2).reshape(batch_size, self.G * self.input_ch_views, pt_num)
            h = parallel_concat([feature, input_views], self.G)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                # h = F.softplus(h, beta=10) if self.softplus else F.relu(h)
                h = F.elu(h) if self.softplus else F.relu(h)
                # h = F.tanh(h) if self.softplus else F.relu(h)

            outputs = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)  # [B, G*C, N]

        outputs = outputs.view([batch_size, self.G, -1, pt_num]).permute(0, 1, 3, 2)  # [B, G, N, C]
        return outputs



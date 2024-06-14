import torch
import torch.nn as nn

from .multi_parallel_nerf import MetaMultiParallelMLP


class ParallelCVAE(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, bottleneck_channels, G, W, D):
        super(ParallelCVAE, self).__init__()
        self.G = G
        self.D = D
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.bottleneck_channels = bottleneck_channels
        self.enc = MetaMultiParallelMLP(
            G, D//2, W, in_channels+cond_channels, bottleneck_channels*2, weight_norm=False, skips=[])
        self.dec = MetaMultiParallelMLP(
            G, D//2, W, cond_channels+bottleneck_channels, out_channels, weight_norm=False, skips=[])

    def encode(self, x, c):
        """
        :param x: [B, G, C]
        :param c: [B, G, C']
        :return:
        """
        h = torch.cat([x, c], dim=-1).unsqueeze(2)
        h, _ = self.enc(h)
        h = h.squeeze(2)
        mu, logvar = torch.split(h, [self.bottleneck_channels, self.bottleneck_channels], dim=-1)
        return mu, logvar

    def decode(self, z, c):
        """
        :param z:
        :param c:
        :return:
        """
        h = torch.cat([z, c], dim=-1).unsqueeze(2)
        h, _ = self.dec(h)
        h = h.squeeze(2)
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_ = self.decode(z, c)
        return x_, mu, logvar



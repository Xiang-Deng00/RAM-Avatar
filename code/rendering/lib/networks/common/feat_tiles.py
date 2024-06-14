import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMap(nn.Module):
    def __init__(self, G=16, num_channels=32, resolution=32):
        super(FeatureMap, self).__init__()
        self.register_parameter(
            'featmap',
            nn.Parameter(torch.randn(G, num_channels, resolution, resolution), requires_grad=True))

    def get_featmap(self):
        return self.featmap

    def forward(self, points, align_corners=True):
        """
        points: [B, G, N, 2]
        """
        points_ = points.permute(1, 0, 2, 3)
        feat = F.grid_sample(self.featmap, points_,
                             mode='bilinear', padding_mode='border', align_corners=align_corners)
        feat = feat.permute(2, 0, 3, 1)     # [G, C, B, N] -> [B, G, N, C]
        return feat


class NodeFeatureTile(nn.Module):
    def __init__(self, G=16, num_channel=32, num_levels=4, max_res=128, align_corners=True):
        super(NodeFeatureTile, self).__init__()
        self.featmap_list = nn.ModuleList(
            [FeatureMap(G, num_channel, max_res // (2**level)) for level in range(num_levels)]
        )
        self.align_corners = align_corners

    def forward(self, points, node_uv_projection):
        """
        points: [B, G, N, 3]
        node_uv_projection: [B, G, 2, 3]

        """
        if len(node_uv_projection.shape) == 4:
            points_uv = torch.einsum('bgnl,bgml->bgnm', points, node_uv_projection)
        elif len(node_uv_projection.shape) == 3:
            points_uv = torch.einsum('bgnl,gml->bgnm', points, node_uv_projection)
        else:
            raise RuntimeError('Invalid shape of node_uv_projection!')
        feats = [featmap(points_uv, self.align_corners) for featmap in self.featmap_list]
        feat = sum(feats)
        return feat

import torch
import torch.nn as nn
import torch.nn.functional as F


class Triplane(nn.Module):
    def __init__(self, num_channels=32, resolution=32):
        super(Triplane, self).__init__()
        self.register_parameter(
            'featmap_xy',
            nn.Parameter(torch.randn(1, num_channels, resolution, resolution), requires_grad=True))
        self.register_parameter(
            'featmap_yz',
            nn.Parameter(torch.randn(1, num_channels, resolution, resolution), requires_grad=True))
        self.register_parameter(
            'featmap_xz',
            nn.Parameter(torch.randn(1, num_channels, resolution, resolution), requires_grad=True))

    def forward(self, points, align_corners=True):
        """
        points: [B, H, C, 3]
        """
        #print('points:', points.shape)
        #print(points[:,0,0])
        batch_size = points.shape[0]
        points_xy = points[:, :, :, 0:2]
        points_yz = points[:, :, :, 1:3]
        points_xz = points[:, :, :, (0,2)]

        feat_xy = F.grid_sample(torch.tanh(self.featmap_xy).expand(batch_size, -1, -1, -1), points_xy,
                                mode='bilinear', padding_mode='border', align_corners=align_corners)
        feat_yz = F.grid_sample(torch.tanh(self.featmap_yz).expand(batch_size, -1, -1, -1), points_yz,
                                mode='bilinear', padding_mode='border', align_corners=align_corners)
        feat_xz = F.grid_sample(torch.tanh(self.featmap_xz).expand(batch_size, -1, -1, -1), points_xz,
                                mode='bilinear', padding_mode='border', align_corners=align_corners)
        feat = torch.cat([feat_xy, feat_yz, feat_xz], dim=1)
        return feat

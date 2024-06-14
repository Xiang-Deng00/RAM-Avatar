import torch
import torch.nn as nn


class AttnNet(nn.Module):
    """
    This attention network is implemented to attached at the end of parallel cVAE
    """
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_head=4, n_attn_layer=4):
        super(AttnNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_attn_layer = n_attn_layer

        self.in_layers_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
        )
        self.attn_enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim),
            norm=nn.LayerNorm(normalized_shape=hidden_dim),
            num_layers=n_attn_layer
        )

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, m=None):
        """
        :param x: [B, G, C]
        :return:
        """
        h = self.in_layers_enc(x)
        h = h.permute(1, 0, 2)
        h = self.attn_enc(h, m)
        h = h.permute(1, 0, 2)
        y = self.out_layer(h)
        return y

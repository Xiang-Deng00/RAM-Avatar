import torch
import torch.nn as nn
import math


class ParallelLinear(nn.Conv1d):
    """
    Use Conv1d to implement parallel MLP
    """
    def __init__(self, G: int, in_features: int, out_features: int, bias: bool = True):
        """

        :param G: number of parallel linear layers
        :param in_features:
        :param out_features:
        :param bias:
        """
        self.G = G
        self.in_features = in_features
        self.out_features = out_features
        super(ParallelLinear, self).__init__(
            in_channels=in_features * G,
            out_channels=out_features * G,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=G,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: [B, G*C, N]
        :return: [B, G*C', N]
        """
        return super(ParallelLinear, self).forward(input)

    def extra_repr(self) -> str:
        return 'parallel_layers={}, in_features={}, out_features={}, bias={}'.format(
            self.G, self.in_features, self.out_features, self.bias is not None
        )


def parallel_concat(tensors: list, n_parallel_group: int):
    """
    :param tensors: list of tensors, each of which has a shape of [B, G*C, N]
    :param n_parallel_group:
    :return: [B, G*C', N]
    """
    batch_size = tensors[0].shape[0]
    point_num = tensors[0].shape[-1]
    assert all([t.shape[0] == batch_size for t in tensors]), 'All tensors should have the same batch size'
    assert all([t.shape[2] == point_num for t in tensors]), 'All tensors should have the same point num'
    assert all([t.shape[1] % n_parallel_group==0 for t in tensors]), 'Invalid tensor channels'

    tensors_ = [
        t.reshape(batch_size, n_parallel_group, -1, point_num) for t in tensors
    ]
    concated = torch.cat(tensors_, dim=2)
    concated = concated.reshape(batch_size, -1, point_num)
    return concated



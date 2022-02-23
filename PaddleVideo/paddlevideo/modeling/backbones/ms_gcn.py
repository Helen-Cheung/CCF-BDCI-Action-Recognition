import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp

import numpy as np

from .utils import activation_factory, k_adjacency, normalize_adjacency_matrix 
from .mlp import MLP
from ..registry import BACKBONES

class MultiScale_GraphConv(nn.Layer):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0.2,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = paddle.Tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = paddle.create_parameter(self.A_powers.shape, self.A_powers.dtype, default_initializer=nn.initializer.Uniform(-1e-6, 1e-6))

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        N, C, T, V = x.shape
        self.A_powers = paddle.to_tensor(self.A_powers, place=x.place)
        A = paddle.to_tensor(self.A_powers, dtype=x.dtype)
        self.A_res = paddle.to_tensor(self.A_res, dtype=x.dtype)
        if self.use_mask:
            A = A + self.A_res
        support = paddlenlp.ops.einsum('vu,nctu->nctv', A, x)
        support = paddle.reshape(support, [N, C, T, self.num_scales, V])
        #support = support.view(N, C, T, self.num_scales, V)
        support = paddle.transpose(support, [0,3,1,2,4])
        support = paddle.reshape(support, [N, self.num_scales*C, T, V])
        #support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out

        


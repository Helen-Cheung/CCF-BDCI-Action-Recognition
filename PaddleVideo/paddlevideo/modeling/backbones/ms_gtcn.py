import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
import numpy as np

from .ms_tcn import MultiScale_TemporalConv as MS_TCN
from .mlp import MLP
from .utils import activation_factory, k_adjacency, normalize_adjacency_matrix 
from ..registry import BACKBONES


class UnfoldTemporalWindows(nn.Layer):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_sizes=[self.window_size, 1],
                                dilations=[self.window_dilation, 1],
                                strides=[self.window_stride, 1],
                                paddings=[self.padding, 0])

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = paddle.reshape(x,[N, C, self.window_size, -1, V])
        x = paddle.transpose(x, [0,1,3,2,4])
        #x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = paddle.reshape(x, [N, C, -1, self.window_size * V])
        #x = x.view(N, C, -1, self.window_size * V)
        return x


class SpatialTemporal_MS_GCN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0.2,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = paddle.Tensor(A_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = paddle.create_parameter(self.A_scales.shape, self.A_scales.dtype, default_initializer=nn.initializer.Uniform(-1e-6, 1e-6))
        else:
            self.A_res = paddle.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='relu')
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='relu')

        self.act = activation_factory(activation)

    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows

        # Build graphs
        self.A_scales = paddle.to_tensor(self.A_scales, place=x.place, dtype=x.dtype)
        self.A_res = paddle.to_tensor(self.A_res, place=x.place, dtype=x.dtype)
        A = self.A_scales + self.A_res

        # Perform Graph Convolution
        res = self.residual(x)
        agg = paddlenlp.ops.einsum('vu,nctu->nctv', A, x)
        agg = paddle.reshape(agg, [N, C, T, self.num_scales, V])
        #agg = agg.view(N, C, T, self.num_scales, V)
        agg = paddle.transpose(agg, [0,3,1,2,4])
        agg = paddle.reshape(agg, [N, self.num_scales*C, T, V])
        #agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)
import math
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import BACKBONES

from .ms_gcn import MultiScale_GraphConv as MS_GCN
from .ms_tcn import MultiScale_TemporalConv as MS_TCN
from .ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from .mlp import MLP
from .utils import activation_factory, import_class, count_params

def iden(x):
    return x

class MS_G3D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = iden
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3D(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = paddle.reshape(x, [N, self.embed_channels_out, -1, self.window_size, V])
        x = self.out_conv(x).squeeze(axis=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.LayerList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

class Graph():
    def __init__(self):

        self.get_edge()
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = self.normalize_adjacency_matrix(self.A_binary)

    def __str__(self):
        return self.A

    def get_edge(self):
        # edge is a list of [child, parent] paris
        self.num_nodes = 25
        self_link = [(i, i) for i in range(self.num_nodes)]
        inward_ori_index = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        self.edges = inward + outward

    def get_adjacency_matrix(self, edges, num_nodes):
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            A[edge] = 1.
        return A
    
    def normalize_adjacency_matrix(self, A):
        node_degrees = A.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
        return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

@BACKBONES.register()
class MSG3D(nn.Layer):
    def __init__(self,
                 num_point=25,
                 num_person=1,
                 num_gcn_scales=18,
                 num_g3d_scales=8,
                 in_channels=2):
        super(MSG3D, self).__init__()

        # load graph
        self.graph = Graph( )
        A_binary = self.graph.A_binary

        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(2, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 2, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = None
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = None
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = None
        self.tcn3 = MS_TCN(c3, c3)
        
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        #self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = paddle.transpose(x, perm=[0, 4, 3, 1, 2])
        x = paddle.reshape(x, [N, M * V * C, T])
        #x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = paddle.reshape(x, [N * M, V, C, T])
        x = paddle.transpose(x, perm=[0, 2, 3, 1]) #N * M, C, T, V
        #x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x))
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x))
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x))
        x = self.tcn3(x)
        x = self.pool(x)  #N * M, C, 1, 1
        
        out = x
        out_channels = out.shape[1]
        #out = paddle.reshape(out, [N, M, out_channels, 1, 1]).mean(axis=1)
        out = out.reshape((N, M, out_channels, -1))
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

       # out = self.fc(out)
        return out

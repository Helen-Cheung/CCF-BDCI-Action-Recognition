import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
#from torch.autograd import Variable
import numpy as np
import math

from .dgcn_graph import Graph
from ..registry import BACKBONES


class TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),   # Conv along the temporal dimension only
            padding=(pad, 0),
            stride=(stride, 1),weight_attr=nn.initializer.KaimingNormal()
        )

        self.bn = nn.BatchNorm2D(out_channels)
        #conv_init(self.conv)
        #bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BiTemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        # NOTE: assuming that temporal convs are shared between node/edge features
        self.tempconv = TemporalConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv(fv), self.tempconv(fe)


class DGNBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, source_M, target_M):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        # Adaptive block with learnable graphs; shapes (V_node, V_edge)
        self.s_M = paddle.to_tensor(source_M.astype('float32'))
        self.source_M = paddle.create_parameter(self.s_M.shape, self.s_M.dtype)
        #self.source_M = nn.Parameter(torch.from_numpy(source_M.astype('float32')))
        self.t_M = paddle.to_tensor(target_M.astype('float32'))
        self.target_M = paddle.create_parameter(self.t_M.shape, self.t_M.dtype)
        #self.target_M = nn.Parameter(torch.from_numpy(target_M.astype('float32')))

        # Updating functions
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)

        self.bn_v = nn.BatchNorm2D(out_channels)
        self.bn_e = nn.BatchNorm2D(out_channels)
        #bn_init(self.bn_v, 1)
        #bn_init(self.bn_e, 1)

        self.relu = nn.ReLU()

    def forward(self, fv, fe):
        # `fv` (node features) has shape (N, C, T, V_node)
        # `fe` (edge features) has shape (N, C, T, V_edge)
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape

        # Reshape for matmul, shape: (N, CT, V)
        fv = fv.reshape((N, -1, V_node))
        fe = fe.reshape((N, -1, V_edge))

        # Compute features for node/edge updates
        fe_in_agg = paddlenlp.ops.einsum('nce,ev->ncv', fe, self.source_M.transpose(0,1))
        fe_out_agg = paddlenlp.ops.einsum('nce,ev->ncv', fe, self.target_M.transpose(0,1))
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)   # Out shape: (N,3,CT,V_nodes)
        fvp = fvp.reshape((N, 3 * C, T, V_node)).transpose((0,2,3,1))   # (N,T,V_node,3C)
        fvp = self.H_v(fvp).transpose((0,3,1,2))    # (N,C_out,T,V_node)
        fvp = self.bn_v(fvp)
        fvp = self.relu(fvp)

        fv_in_agg = paddlenlp.ops.einsum('ncv,ve->nce', fv, self.source_M)
        fv_out_agg = paddlenlp.ops.einsum('ncv,ve->nce', fv, self.target_M)
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)   # Out shape: (N,3,CT,V_edges)
        fep = fep.reshape(N, 3 * C, T, V_edge).transpose((0,2,3,1))   # (N,T,V_edge,3C)
        fep = self.H_e(fep).transpose((0,3,1,2))    # (N,C_out,T,V_edge)
        fep = self.bn_e(fep)
        fep = self.relu(fep)
        return fvp, fep


class GraphTemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.dgn = DGNBlock(in_channels, out_channels, source_M, target_M)
        self.tcn = BiTemporalConv(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = BiTemporalConv(in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv_res, fe_res = self.residual(fv, fe)
        fv, fe = self.dgn(fv, fe)
        fv, fe = self.tcn(fv, fe)
        fv += fv_res
        fe += fe_res
        return self.relu(fv), self.relu(fe)

@BACKBONES.register()
class DGCN(nn.Layer):
    def __init__(self, num_point=25, num_person=1, in_channels=2):
        super(DGCN, self).__init__()


        self.graph = Graph()

        source_M, target_M = self.graph.source_M, self.graph.target_M
        self.data_bn_v = nn.BatchNorm1D(num_person * in_channels * num_point)
        self.data_bn_e = nn.BatchNorm1D(num_person * in_channels * num_point)

        self.l1 = GraphTemporalConv(3, 64, source_M, target_M, residual=False)
        self.l2 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l3 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l4 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l5 = GraphTemporalConv(64, 128, source_M, target_M, stride=2)
        self.l6 = GraphTemporalConv(128, 128, source_M, target_M)
        self.l7 = GraphTemporalConv(128, 128, source_M, target_M)
        self.l8 = GraphTemporalConv(128, 256, source_M, target_M, stride=2)
        self.l9 = GraphTemporalConv(256, 256, source_M, target_M)
        self.l10 = GraphTemporalConv(256, 256, source_M, target_M)

        #self.fc = nn.Linear(256 * 2, num_class)

        #nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        #bn_init(self.data_bn_v, 1)
        #bn_init(self.data_bn_e, 1)

    def forward(self, fv, fe):
        N, C, T, V_node, M = fv.shape
        _, _, _, V_edge, _ = fe.shape

        # Preprocessing
        fv = fv.transpose((0, 4, 3, 1, 2)).reshape((N, M * V_node * C, T))
        fv = self.data_bn_v(fv)
        fv = fv.reshape((N, M, V_node, C, T)).transpose((0, 1, 3, 4, 2)).reshape((N * M, C, T, V_node))

        fe = fe.transpose((0, 4, 3, 1, 2)).reshape((N, M * V_edge * C, T))
        fe = self.data_bn_e(fe)
        fe = fe.reshape((N, M, V_edge, C, T)).transpose((0, 1, 3, 4, 2)).reshape((N * M, C, T, V_edge))

        fv, fe = self.l1(fv, fe)
        fv, fe = self.l2(fv, fe)
        fv, fe = self.l3(fv, fe)
        fv, fe = self.l4(fv, fe)
        fv, fe = self.l5(fv, fe)
        fv, fe = self.l6(fv, fe)
        fv, fe = self.l7(fv, fe)
        fv, fe = self.l8(fv, fe)
        fv, fe = self.l9(fv, fe)
        fv, fe = self.l10(fv, fe)

        # Shape: (N*M,C,T,V), C is same for fv/fe
        out_channels = fv.shape[1]

        # Performs pooling over both nodes and frames, and over number of persons
        fv = fv.reshape((N, M, out_channels, -1)).mean(3).mean(1)
        fe = fe.reshape((N, M, out_channels, -1)).mean(3).mean(1)

        # Concat node and edge features
        out = paddle.concat((fv, fe), axis=-1)

        return out

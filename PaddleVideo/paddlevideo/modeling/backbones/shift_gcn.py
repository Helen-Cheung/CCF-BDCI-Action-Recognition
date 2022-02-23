import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
import numpy as np
import math

from ..registry import BACKBONES
from cuda.shift import Shift
from .stgcn import Graph



class tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), weight_attr=nn.initializer.KaimingNormal())

        self.bn = nn.BatchNorm2D(out_channels, weight_attr=nn.initializer.Constant(value=1), bias_attr=nn.initializer.Constant(value=0))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2D(in_channels)
        self.bn2 = nn.BatchNorm2D(in_channels, weight_attr=nn.initializer.Constant(value=1), bias_attr=nn.initializer.Constant(value=0))

        self.relu = nn.ReLU()
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2D(in_channels, out_channels, 1, weight_attr=nn.initializer.KaimingNormal())

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, weight_attr=nn.initializer.KaimingNormal()),
                nn.BatchNorm2D(out_channels, weight_attr=nn.initializer.Constant(value=1), bias_attr=nn.initializer.Constant(value=0))
            )
        else:
            self.down = lambda x: x
        
        self.L_W = paddle.zeros(in_channels, out_channels)
        self.Linear_weight = paddle.create_parameter(self.L_W.shape, self.L_W.dtype, default_initializer=nn.initializer.Normal(mean=0.0, std=math.sqrt(1.0/out_channels)))
        #self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        #nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.L_b = paddle.zeros(1,1,out_channels)
        self.Linear_bias = paddle.create_parameter(self.L_b.shape, self.L_b.dtype, default_initializer=nn.initializer.Constant(value=1))
        #self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        #nn.init.constant(self.Linear_bias, 0)

        self.F_M = paddle.ones(1,25,in_channels)
        self.Feature_Mask = paddle.create_parameter(self.F_M.shape, self.F_M.dtype, default_initializer=nn.initializer.Constant(value=0))
        #self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        #nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1D(25*out_channels)
        self.relu = nn.ReLU()

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
         #       conv_init(m)
         #   elif isinstance(m, nn.BatchNorm2d):
         #       bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.s_in = paddle.to_tensor(index_array)
        self.shift_in = paddle.create_parameter(self.s_in.shape, self.s_in.dtype)
        #self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.s_out = paddle.to_tensor(index_array)
        self.shift_out = paddle.create_parameter(self.s_out.shape, self.s_out.dtype)
        #self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.shape

        x = x0.transpose((0,2,3,1)) #N, T, V, C

        # shift1
        x = x.reshape((n*t,v*c))
        x = paddle.index_select(x, self.shift_in, axis=1)
        x = x.reshape((n*t,v,c))
        x = x * (paddle.tanh(self.Feature_Mask)+1)

        x = paddlenlp.ops.einsum('nwc,cd->nwd', (x, self.Linear_weight)) # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.reshape((n*t,-1)) 
        x = paddle.index_select(x, self.shift_out, axis=1)
        x = self.bn(x)
        x = x.reshape((n,t,v,self.out_channels)).transpose((0,3,1,2)) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


@BACKBONES.register()
class SGCN(nn.Layer):
    def __init__(self, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=2, layout='fsd10',
                 strategy='spatial'):
        super(SGCN, self).__init__()

        # load graph
        self.graph = Graph(
            layout=layout,
            strategy=strategy,
        )
        A = paddle.to_tensor(self.graph.A, dtype='float32')
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point, weight_attr=nn.initializer.Constant(value=1), bias_attr=nn.initializer.Constant(value=0))

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)
        
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        #self.fc = nn.Linear(256, num_class)
        #nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        #bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 3, 1, 2)).reshape((N, M * V * C, T)) 
        x = self.data_bn(x)
        x = x.reshape((N, M, V, C, T)).transpose((0, 1, 3, 4, 2)).reshape((N * M, C, T, V)) 

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        x = self.pool(x)
        c_new = x.shape[1]
        x = paddle.reshape(x, (N, M, c_new, 1, 1)).mean(axis=1)  # N,C,1,1

        return x
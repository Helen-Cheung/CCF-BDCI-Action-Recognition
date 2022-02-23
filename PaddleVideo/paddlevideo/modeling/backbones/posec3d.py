import paddle
import paddle.nn as nn
import collections

from itertools import repeat
from ..registry import BACKBONES


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class BasicBlock3d(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 **kwargs):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)
        
        self.conv1 = nn.Conv3D(inplanes, planes, conv1_kernel_size, stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s), padding=conv1_padding, dilation=(1, dilation, dilation), bias_attr=False)

        self.bn1 = nn.BatchNorm3D(planes)
        
        self.conv2 = nn.Conv3D(planes, planes* self.expansion, conv2_kernel_size, stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s), padding=conv2_padding, bias_attr=False)
        self.bn2 = nn.BatchNorm3D(planes* self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):

        def _inner_forward(x):

            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1'):
        super().__init__()

        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.inflate_style = inflate_style

        
        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = nn.Conv3D(inplanes, planes, conv1_kernel_size, stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s), padding=conv1_padding, bias_attr=False)
        self.bn1 = nn.BatchNorm3D(planes)

        self.conv2 = nn.Conv3D(planes, planes, conv2_kernel_size, stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s), padding=conv2_padding, dilation=(1, dilation, dilation), bias_attr=False)
        self.bn2 = nn.BatchNorm3D(planes)

        self.conv3 = nn.Conv3D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm3D(planes * self.expansion)


        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):

        def _inner_forward(x):

            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class ResNet3d(nn.Layer):
    """ResNet 3d backbone."""

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 stage_blocks=None,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(3, 7, 7),
                 conv1_stride_s=2,
                 conv1_stride_t=1,
                 pool1_stride_s=2,
                 pool1_stride_t=1,
                 with_pool1=True,
                 with_pool2=True,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 **kwargs):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2

        self.stage_inflations = _ntuple(num_stages)(inflate)

        self.inflate_style = inflate_style


        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       **kwargs):
        """Build residual layer for ResNet3D."""

        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks

        assert len(inflate) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:            
            downsample = nn.Sequential(
                nn.Conv3D(inplanes, planes * block.expansion, kernel_size=1, stride=(temporal_stride, spatial_stride, spatial_stride), bias_attr=False),
                nn.BatchNorm3D(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    **kwargs))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""

        self.conv1 = nn.Sequential(
            nn.Conv3D(self.in_channels, self.base_channels, kernel_size=self.conv1_kernel, stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),  padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]), bias_attr=False),
            nn.BatchNorm3D(self.base_channels),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool3D(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s,
                    self.pool1_stride_s),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3D(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = x.transpose((0,3,2,1,4))
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d."""

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:

            self.conv1_lateral = nn.Conv3D(self.inplanes // self.channel_ratio, self.inplanes * 2 // self.channel_ratio, 
                kernel_size=(fusion_kernel, 1, 1), stride=(self.speed_ratio, 1, 1), padding=((fusion_kernel - 1) // 2, 0, 0), bias_attr=False)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                setattr(
                    self, lateral_name,
                    nn.Conv3D(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias_attr=False
                        ))
                self.lateral_connections.append(lateral_name)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1'):
        """Build residual layer for Slowfast."""

        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks

        assert len(inflate) == blocks 
        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0
        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv3D(inplanes + lateral_inplanes, planes * block.expansion, kernel_size=1, 
                stride=(temporal_stride, spatial_stride, spatial_stride), bias_attr=False),
                nn.BatchNorm3D(planes * block.expansion)
            )
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style))
        inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style))

        return nn.Sequential(*layers)


@BACKBONES.register()
class Posec3d(ResNet3dPathway):

    def __init__(self,
                 depth=101,
                 in_channels=25,
                 base_channels=32,
                 num_stages = 3,
                 out_indices = (2, ),
                 stage_blocks = (4, 6, 3),
                 conv1_stride_s=1,
                 pool1_stride_s=1,
                 inflate = (0, 1, 1),
                 spatial_strides = (2,2,2),
                 temporal_strides=(1, 1, 2),
                 dilations = (1, 1, 1),
                 lateral=False,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 with_pool2=False,
                 **kwargs):
        super().__init__(
            depth=depth,
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages = num_stages,
            out_indices = out_indices,
            stage_blocks = stage_blocks,
            conv1_stride_s=conv1_stride_s,
            pool1_stride_s=pool1_stride_s,
            inflate = inflate,
            spatial_strides = spatial_strides,
            temporal_strides=temporal_strides,
            dilations = dilations,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral


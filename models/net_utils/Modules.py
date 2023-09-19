import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models_.net_utils.STGCN_block import st_gcn_block
from models_.net_utils.tgcn import ConvTemporalGraphical
from models_.net_utils.graph import Graph
class UP_new(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_c, out_c,v,up_type):
        super(UP_new, self).__init__()
        assert up_type in ['bilinear','nearest','transpose']
        self.up_type = up_type
        self.conv0 = nn.Conv2d(in_c,out_c,kernel_size=(1,1)) # 调整下采样特征通道
        self.conv_out = double_conv(out_c + out_c, out_c)
        self.trans_conv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=(2,1), stride=(2,1)
        )
        self.relu = torch.nn.ReLU()
        self.v = v

    def forward(self, down_fea, up_fea):
        # 下采样特征和上采样特征concat
        # 做一个1D卷积调整一下通道
        # 做一个卷积到c=1
        # sigmoid激活后乘下采样特征再次和上采样拼在一起。这种方法是上采样和下采样特征拼起来发现有些地方不对就加重下采样特征权重，
        # 重新和当前上采样拼起来。也可以上层产生attention指导下层
        if self.up_type != 'transpose':
            up_fea = nn.functional.interpolate(up_fea, (down_fea.shape[2], down_fea.shape[3]),mode=self.up_type)

        else:
            up_fea = self.trans_conv(up_fea)
            diff = torch.tensor([down_fea.size()[2] - up_fea.size()[2]])
            up_fea = F.pad(up_fea, (0, 0, diff // 2, diff - diff // 2))
        down_fea = self.conv0(down_fea)

        fuse_fea = torch.cat((down_fea,up_fea),dim=1)
        out = self.conv_out(fuse_fea)
        return out

class UP(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_c, out_c, v, up_type, with_att, att_type=None):
        super(UP, self).__init__()
        assert up_type in ['bilinear', 'nearest', 'transpose']
        self.up_type = up_type
        if with_att:
            assert att_type is not None
            assert att_type in ['max', 'avg', 'all']
            self.att_type = att_type
        self.with_att = with_att
        self.conv0 = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))  # 调整下采样特征通道
        self.conv_out = double_conv(out_c + out_c, out_c)
        self.conv_att = nn.Sequential(
            _1DConv(out_c + out_c, out_c),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            _1DConv(out_c, 1)
        )
        self.trans_conv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=(2, 1), stride=(2, 1)
        )
        self.relu = torch.nn.ReLU()
        self.v = v

    def forward(self, down_fea, up_fea):
        # 下采样特征和上采样特征concat
        # 做一个1D卷积调整一下通道
        # 做一个卷积到c=1
        # sigmoid激活后乘下采样特征再次和上采样拼在一起。这种方法是上采样和下采样特征拼起来发现有些地方不对就加重下采样特征权重，
        # 重新和当前上采样拼起来。也可以上层产生attention指导下层
        if self.up_type != 'transpose':
            up_fea = nn.functional.interpolate(up_fea, (down_fea.shape[2], down_fea.shape[3]), mode=self.up_type)

        else:
            up_fea = self.trans_conv(up_fea)
            diff = torch.tensor([down_fea.size()[2] - up_fea.size()[2]])
            up_fea = F.pad(up_fea, (0, 0, diff // 2, diff - diff // 2))
        down_fea = self.conv0(down_fea)
        if self.with_att:
            fuse_fea = torch.cat((down_fea, up_fea), dim=1)
            att = self.conv_att(fuse_fea)
            if self.att_type == 'max':
                att = F.max_pool2d(att, kernel_size=(1, self.v))
            elif self.att_type == 'avg':
                att = F.avg_pool2d(att, kernel_size=(1, self.v))
            else:
                att = att
            att = torch.sigmoid(att)
            down_fea = down_fea * att

        fuse_fea = torch.cat((down_fea, up_fea), dim=1)
        out = self.conv_out(fuse_fea)
        return out


class ConvAtt(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_c, out_c,v,with_att,att_type=None):
        super(ConvAtt, self).__init__()
        assert att_type is not None
        assert att_type in ['max','avg','all']
        self.att_type = att_type
        self.conv_att = nn.Sequential(
            nn.Conv2d(in_c, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            _1DConv(128, 1)
        )
        self.relu = torch.nn.ReLU()
        self.v = v

    def forward(self, fea):
        att = self.conv_att(fea)
        if self.att_type == 'max':
            att = F.max_pool2d(att, kernel_size=(1, self.v))
        elif self.att_type == 'avg':
            att = F.avg_pool2d(att, kernel_size=(1, self.v))
        else:
            att = att
        return att.squeeze(-1)
class _1DConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_1DConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch,pooling_type = 'max'):
        super(down, self).__init__()
        if pooling_type == 'avg':
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d((2, 1)), double_conv(in_ch, out_ch))
        else:
            self.pool_conv = nn.Sequential(
                nn.MaxPool2d((2, 1)), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.pool_conv(x)
        return x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.up = nn.ConvTranspose1d(
            in_channels // 2, in_channels // 2, kernel_size=2, stride=2
        )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = nn.functional.interpolate(x1, (x2.shape[2], x2.shape[3]))
        else:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, v):
        super(outconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1))
        self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_ch)
        self.v = v

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, kernel_size=(1, self.v)).squeeze(-1)
        x = self.conv2(x)
        return x

class c2f_st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` formatz
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 A=None,
                 dilation=1,
                 residual=True):
        super(c2f_st_gcn_block, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation * (kernel_size[0] - 1)) / 2)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0]),
                stride=stride,
                padding=pad,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = self.relu(x)
        x = x + res
        return x, A
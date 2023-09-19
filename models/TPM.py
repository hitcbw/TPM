import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.net_utils.STGCN_block import st_gcn_block
from models.net_utils.tgcn import ConvTemporalGraphical
from models.net_utils.graph import Graph
from models.net_utils.Modules import UP_new, down, outconv, UP


# 试一下sample rate对hugadb的影响
class Model(nn.Module):
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

    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.channel
        feature_dim = args.feat_dim
        filters = args.spatial_filters
        edge_importance_weighting = args.edge_importance_weighting
        n_classes = args.num_classes
        graph_args = {'layout': args.graph_layout, 'strategy': args.graph_strategy}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = down(64, 64)
        self.down2 = down(64, 64)
        self.down3 = down(64, 64)
        self.down4 = down(64, 64)
        self.down5 = down(64, 64)
        self.down6 = down(64, 64)
        self.UP0 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.UP1 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.UP2 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.UP3 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.UP4 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.UP5 = UP(64, 64, v=self.A.shape[1], up_type='bilinear', with_att=False, att_type='avg')
        self.outcc0 = outconv(64, n_classes,v=self.A.shape[1])
        self.outcc1 = outconv(64, n_classes,v=self.A.shape[1])
        self.outcc2 = outconv(64, n_classes,v=self.A.shape[1])
        self.outcc3 = outconv(64, n_classes, v=self.A.shape[1])
        self.outcc4 = outconv(64, n_classes, v=self.A.shape[1])
        self.outcc5 = outconv(64, n_classes, v=self.A.shape[1])
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)

        # ---
        # initialize parameters for edge importance weighting
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True),
            st_gcn_block(filters, filters, kernel_size, 1, A=A, residual=True)
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.cls_head = nn.Conv1d(filters, feature_dim, kernel_size=1)

    def ensemble(self, logits):
        outputs = []
        for idx, logit in enumerate(logits):
            prob = F.softmax(logit,dim=1)
            prob = F.interpolate(prob, size=logits[-1].shape[-1],mode='linear').unsqueeze(0)
            outputs.append(prob)
        outputs = torch.cat(outputs)
        output = torch.sum(outputs,dim=0) / len(logits)
        return output

    def get_new_smooth_label(self, logits):
        outputs = []
        for idx, logit in enumerate(logits):
            prob = F.softmax(logit, dim=1)
            prob = F.interpolate(prob, size=logits[-1].shape[-1]).unsqueeze(0)
            outputs.append(prob)
        outputs = torch.cat(outputs)
        return outputs

    def forward(self, x, mask):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # adjust channels
        x1 = self.inconv(x)

        # down conv
        # x: 128   T
        x1,  _ = self.st_gcn_networks[0](x1, self.A * self.edge_importance[0])
        x2 = self.down1(x1)
        # x: 128   T/2
        x2,  _ = self.st_gcn_networks[1](x2, self.A * self.edge_importance[1])
        x3 = self.down2(x2)
        # x: 128   T/4
        x3,  _ = self.st_gcn_networks[2](x3, self.A * self.edge_importance[2])
        x4 = self.down3(x3)
        # x: 64   T/8
        x4,  _ = self.st_gcn_networks[3](x4, self.A * self.edge_importance[3])
        x5 = self.down4(x4)
        # x: 64   T/16
        x5, _ = self.st_gcn_networks[4](x5, self.A * self.edge_importance[4])
        x6 = self.down5(x5)

        x6, _ = self.st_gcn_networks[5](x6, self.A * self.edge_importance[5])
        x7 = self.down6(x6)

        x7, _ = self.st_gcn_networks[6](x7, self.A * self.edge_importance[6])

        x = self.UP0(x6, x7)
        # x: 64   T/ 16

        x, _ = self.st_gcn_networks[7](x, self.A * self.edge_importance[7])
        y0 = self.outcc0(F.relu(x))

        x = self.UP1(x5, x)
        # x: 64   T/ 16
        x, _ = self.st_gcn_networks[8](x, self.A * self.edge_importance[8])
        # x: 64    T/32
        y1 = self.outcc1(F.relu(x))

        x = self.UP2(x4, x)
        x,  _ = self.st_gcn_networks[9](x, self.A * self.edge_importance[9])
        # x: 64   T/16
        y2 = self.outcc2(F.relu(x))

        x = self.UP3(x3, x)
        x,  _ = self.st_gcn_networks[10](x, self.A * self.edge_importance[10])
        # x: 64   T/8
        y3 = self.outcc3(F.relu(x))

        x = self.UP4(x2, x)
        x, _ = self.st_gcn_networks[11](x, self.A * self.edge_importance[11])
        # x: 64   T/8
        y4 = self.outcc4(F.relu(x))

        x = self.UP5(x1, x)
        x, _ = self.st_gcn_networks[12](x, self.A * self.edge_importance[12])
        # x: 64   T/8
        y5 = self.outcc5(F.relu(x))

        ensemble = self.ensemble([y0,y1,y2,y3,y4,y5])
        smooth_labels = self.get_new_smooth_label([y0,y1,y2,y3,y4,y5])
        return [y0,y1,y2,y3,y4,y5,ensemble,smooth_labels]





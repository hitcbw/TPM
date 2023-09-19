import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from scipy.signal import argrelmax

from models_.net_utils.tgcn import ConvTemporalGraphical
from models_.net_utils.graph import Graph
# from .modules import GRUPrediction, FCPrediction, TCNPrediction, ASFormerPrediction, ASRFPrediction, BiLSTMPrediction


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

    def __init__(self, in_channels=3, num_class=2, dil=[1,2,4,8,16], filters=64,
                 edge_importance_weighting=True, **kwargs):
        super(Model, self).__init__()
        graph_args = {'layout': 'lara', 'strategy': 'spatial'}
        # load graph
        # print('--------')
        # print(graph_args)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[0], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[1], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[2], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[3], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[4], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[5], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[6], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[7], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[8], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[9], residual=True),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.conv_out = nn.Conv1d(filters, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        x = self.conv_1x1(x)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)
        out = self.conv_out(x)
        return out


class st_gcn(nn.Module):
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
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation*(kernel_size[0]-1))/2)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=(pad, 0),
                dilation=(dilation, 1),
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

class MultiStageModel(nn.Module):
    def __init__(self, dil, num_layers_R, num_R, num_f_maps, dim, num_classes, head = ''):
        super(MultiStageModel, self).__init__()

        self.stream = Model(in_channels=dim, num_class=num_classes, filters=num_f_maps, dil=dil)
        self.head = head
        if self.head == 'tcn':
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers_R, num_f_maps, num_classes, num_classes, dil)) for s in range(num_R-1)])
        elif self.head == 'FC':
            self.stages = FCPrediction(num_classes,num_classes)
        elif self.head == 'asformer':
            self.stages = ASFormerPrediction(num_classes,num_classes)
        elif self.head == 'gru':
            self.stages = GRUPrediction(num_classes,num_classes,64,1)
        elif self.head == 'asrf':
            self.stages = ASRFPrediction(dim,num_f_maps,num_classes,4,10,4,4)
        elif self.head == 'bilstm':
            self.stages = BiLSTMPrediction(num_classes)
        else:
            return

    def forward(self, x, mask):
        out = self.stream(x) * mask[:, 0:1, :]
        if self.head in ['FC','gru']:
            outputs = self.stages(out).unsqueeze(0)
        elif self.head in ['asformer']:
            outputs = self.stages(out, mask)
        elif self.head in ['bilstm']:
            outputs = self.stages(out,mask).unsqueeze(0)
        else:
            outputs = out.unsqueeze(0)
            for s in self.stages:
                out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
                outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dil):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dil[i], num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, mask):
        out = self.conv_dilated(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x
        return out * mask[:, 0:1, :]
def decide_boundary_prob_with_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Decide action boundary probabilities based on adjacent frame similarities.
    Args:
        x: frame-wise video features (N, C, T)
    Return:
        boundary: action boundary probability (N, 1, T)
    """
    device = x.device

    # gaussian kernel.
    diff = x[0, :, 1:] - x[0, :, :-1]
    similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * 1.0))

    # define action starting point as action boundary.
    start = torch.ones(1).float().to(device)
    boundary = torch.cat([start, similarity])
    boundary = boundary.view(1, 1, -1)
    return boundary


class PostProcessor(object):
    def __init__(
        self,
        name: str,
        boundary_th: int = 0.7,
        theta_t: int = 15,
        kernel_size: int = 15,
    ) -> None:
        self.func = {
            "refinement_with_boundary": self._refinement_with_boundary,
            "relabeling": self._relabeling,
            "smoothing": self._smoothing,
        }
        assert name in self.func

        self.name = name
        self.boundary_th = boundary_th
        self.theta_t = theta_t
        self.kernel_size = kernel_size

        if name == "smoothing":
            self.filter = GaussianSmoothing(self.kernel_size)

    def _is_probability(self, x: np.ndarray) -> bool:
        assert x.ndim == 3

        if x.shape[1] == 1:
            # sigmoid
            if x.min() >= 0 and x.max() <= 1:
                return True
            else:
                return False
        else:
            # softmax
            _sum = np.sum(x, axis=1).astype(np.float32)
            _ones = np.ones_like(_sum, dtype=np.float32)
            return np.allclose(_sum, _ones)

    def _convert2probability(self, x: np.ndarray) -> np.ndarray:
        """
        Args: x (N, C, T)
        """
        assert x.ndim == 3

        if self._is_probability(x):
            return x
        else:
            if x.shape[1] == 1:
                # sigmoid
                prob = 1 / (1 + np.exp(-x))
            else:
                # softmax
                prob = np.exp(x) / np.sum(np.exp(x), axis=1)
            return prob.astype(np.float32)

    def _convert2label(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 or x.ndim == 3

        if x.ndim == 2:
            return x.astype(np.int64)
        else:
            if not self._is_probability(x):
                x = self._convert2probability(x)

            label = np.argmax(x, axis=1)
            return label.astype(np.int64)

    def _refinement_with_boundary(
        self,
        outputs: np.array,
        boundaries: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Get segments which is defined as the span b/w two boundaries,
        and decide their classes by majority vote.
        Args:
            outputs: numpy array. shape (N, C, T)
                the model output for frame-level class prediction.
            boundaries: numpy array.  shape (N, 1, T)
                boundary prediction.
            masks: np.array. np.bool. shape (N, 1, T)
                valid length for each video
        Return:
            preds: np.array. shape (N, T)
                final class prediction considering boundaries.
        """

        preds = self._convert2label(outputs)
        boundaries = self._convert2probability(boundaries)

        for i, (output, pred, boundary, mask) in enumerate(
            zip(outputs, preds, boundaries, masks)
        ):
            boundary = boundary[mask]
            idx = argrelmax(boundary, threshold=self.boundary_th)

            # add the index of the last action ending
            T = pred.shape[0]
            idx.append(T)

            # majority vote
            for j in range(len(idx) - 1):
                count = np.bincount(pred[idx[j] : idx[j + 1]])
                modes = np.where(count == count.max())[0]
                if len(modes) == 1:
                    mode = modes
                else:
                    if outputs.ndim == 3:
                        # if more than one majority class exist
                        prob_sum_max = 0
                        for m in modes:
                            prob_sum = output[m, idx[j] : idx[j + 1]].sum()
                            if prob_sum_max < prob_sum:
                                mode = m
                                prob_sum_max = prob_sum
                    else:
                        # decide first mode when more than one majority class
                        # have the same number during oracle experiment
                        mode = modes[0]

                preds[i, idx[j] : idx[j + 1]] = mode

        return preds

    def _relabeling(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """

        preds = self._convert2label(outputs)

        for i in range(preds.shape[0]):
            # shape (T,)
            last = preds[i][0]
            cnt = 1
            for j in range(1, preds.shape[1]):
                if last == preds[i][j]:
                    cnt += 1
                else:
                    if cnt > self.theta_t:
                        cnt = 1
                        last = preds[i][j]
                    else:
                        preds[i][j - cnt : j] = preds[i][j - cnt - 1]
                        cnt = 1
                        last = preds[i][j]

            if cnt <= self.theta_t:
                preds[i][j - cnt : j] = preds[i][j - cnt - 1]

        return preds

    def _smoothing(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """

        outputs = self._convert2probability(outputs)
        outputs = self.filter(torch.Tensor(outputs)).numpy()

        preds = self._convert2label(outputs)
        return preds

    def __call__(self, outputs, **kwargs: np.ndarray) -> np.ndarray:
        preds = self.func[self.name](outputs, **kwargs)
        return preds

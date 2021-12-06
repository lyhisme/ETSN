import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ed_tcn import EDTCN

class MultiStageTCN(nn.Module):

    def __init__(self, in_channel, n_classes, stages,
                 n_features=64, dilated_n_layers=10, kernel_size=15):
    
        super().__init__()
        if stages[0] == 'dilated':
            self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, dilated_n_layers)
        elif stages[0] == 'etspnet':
            self.stage1 = Single_ETSPNet(in_channel, n_features, n_classes, dilated_n_layers) 
        elif stages[0] == 'ed':
            self.stage1 = EDTCN(in_channel, n_classes)
        else:
            print("Invalid values as stages in Mixed Multi Stage TCN")
            sys.exit(1)

        if len(stages) == 1:
            self.stages = None
        else:
            self.stages = []
            for stage in stages[1:]:
                if stage == 'dilated':
                    self.stages.append(SingleStageTCN(
                        n_classes, n_features, n_classes, dilated_n_layers))
                elif stage == 'ed':
                    self.stages.append(
                        EDTCN(n_classes, n_classes, kernel_size=kernel_size))
                else:
                    print("Invalid values as stages in Mixed Multi Stage TCN")
                    sys.exit(1)
            self.stages = nn.ModuleList(self.stages)

    def forward(self, x):

        if self.training:

            # # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
                    outputs.append(out)
            return outputs
        else:

            # for evaluation
            out = self.stage1(x)
            if self.stages is not None:
                for stage in self.stages:
                    out = stage(F.softmax(out, dim=1))
            return out


class Single_ETSPNet(nn.Module):

    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.act = nn.PReLU(n_features*n_layers)  
        self.conv1_1 = nn.Conv1d(n_features*n_layers, n_features, 1)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
            out = []
            in_feat = self.conv_in(x)
            for layer in self.layers:
                out.append(layer(in_feat + out[-1] if len(out) != 0 else in_feat))

            out_cat = self.act(torch.cat(out, dim=1))
            output = self.conv1_1(out_cat)
            output = self.conv_out(output + in_feat)

            return output


class SingleStageTCN(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):

        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    # Originally written by yabufarha
    # https://github.com/yabufarha/ms-tcn/blob/master/model.py

    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out
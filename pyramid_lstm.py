import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import namedtuple
import numpy as np

DirectedCLSTM = namedtuple('DirectedCLSTM', ['c_lstm', 'dir'])


class GateFilter(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GateFilter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.bias = Parameter(torch.Tensor(out_channels))

    def forward(self, x, h, use_sigmoid = True):
        x = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))        
        if use_sigmoid:
            return torch.sigmoid(x + h + self.bias)
        else:
            return torch.tanh(x + h + self.bias)

class CLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(CLSTM, self).__init__()
        self.i_gate = GateFilter(1,1)
        self.f_gate = GateFilter(1,1)
        self.c_gate = GateFilter(1,1)
        self.o_gate = GateFilter(1,1)

    def get_dim_from_direction(self, direction):
        return int(direction/2) + 2

    def forward(self, x, direction):        
        shape = x.shape
        dim = self.get_dim_from_direction(direction)
        positions = np.arange(shape[dim])
        if direction % 2 != 0:
            positions = np.flip(positions)
        
        h = torch.zeros_like(x, dtype=torch.float)
        if torch.cuda.is_available:
            h = h.cuda()
        x_chunked = x.chunk(len(positions), dim=dim)
        x_chunked = [t.squeeze(dim = dim) for t in x_chunked]
        h_chunked = h.chunk(len(positions), dim=dim)
        h_chunked = [t.squeeze(dim = dim) for t in h_chunked]        

        for idx, (pos) in enumerate(positions):             
            if idx == 0:
                h_prev = h_chunked[pos]
            else:
                h_prev = h_chunked[positions[idx-1]]
            i_now = self.i_gate(x_chunked[pos], h_prev)
            f_now = self.f_gate(x_chunked[pos], h_prev)
            c_now = self.c_gate(x_chunked[pos], h_prev, use_sigmoid = False)
            if idx > 0:
                c_now = c_now*i_now + c_prev*i_now
            else:
                c_now = c_now*i_now
            c_prev = c_now.clone()
            o_now = self.o_gate(x_chunked[pos], h_prev)
            h_chunked[pos] = o_now * torch.tanh(c_now)
        
        h = torch.stack(h_chunked, dim = dim)
        return h

class PyramidLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(PyramidLSTM, self).__init__()
        self.lstm_list = [CLSTM() for i in range(6)]                
        self.lstm_list = nn.ModuleList(self.lstm_list)
        # self.conv1 = nn.Conv3d(1, 64, kernel_size = 3, stride=2)
        # self.conv2 = nn.Conv3d(64, 128, kernel_size = 3, stride=2)
        # self.conv3 = nn.Conv3d(128, 128, kernel_size = 3, stride=2)
        self.fc = nn.Linear(15680, 10)
        ## for each direction, fist do convolution and then use the result for gates and memories

    def forward(self, x):        
        h_list = [self.lstm_list[i](x, i) for i in range(6)]
        out = torch.mean(torch.stack(h_list), dim = 0)
        # out = F.relu(self.conv1(out))
        # out = F.relu(self.conv2(out))
        # out = F.relu(self.conv3(out))
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    

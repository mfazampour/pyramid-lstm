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
        self.conv_x = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_h = nn.Conv2d(1, out_channels, 3, padding=1)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x, h, use_sigmoid = True):
        x = F.relu(self.conv_x(x))
        h = F.relu(self.conv_h(h))        
        if use_sigmoid:            
            return torch.sigmoid(x + h + self.bias.view(1,-1,1,1).expand_as(x))
        else:
            return torch.tanh(x + h + self.bias.view(1,-1,1,1).expand_as(x))

class CLSTM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super(CLSTM, self).__init__()
        self.i_gate = GateFilter(in_channels,out_channels)
        self.f_gate = GateFilter(in_channels,out_channels)
        self.c_gate = GateFilter(in_channels,out_channels)
        self.o_gate = GateFilter(in_channels,out_channels)
        self.squeeze_h = nn.Conv2d(out_channels, 1, 3, padding=1)
        self.out_channels = out_channels

    def get_dim_from_direction(self, direction):
        return int(direction/2) + 2

    def forward(self, x, direction):        
        shape = x.shape
        dim = self.get_dim_from_direction(direction)
        positions = np.arange(shape[dim])
        if direction % 2 != 0:
            positions = np.flip(positions)
        
        shape = list(x.shape)
        shape[1] = self.out_channels
        h = torch.zeros(shape, dtype=torch.float)
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
            h_prev = self.squeeze_h(h_prev)
            i_now = self.i_gate(x_chunked[pos], h_prev)
            f_now = self.f_gate(x_chunked[pos], h_prev)
            c_now = self.c_gate(x_chunked[pos], h_prev, use_sigmoid = False)
            o_now = self.o_gate(x_chunked[pos], h_prev)
            if idx > 0:
                c_now = c_now*i_now + c_prev*f_now
            else:
                c_now = c_now*i_now
            c_prev = c_now.clone()
            h_chunked[pos] = o_now * torch.tanh(c_now)
        
        h = torch.stack(h_chunked, dim = dim)
        return h

class PyramidLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(PyramidLSTM, self).__init__()
        self.lstm_list1 = [CLSTM(out_channels=4) for i in range(6)]                
        self.lstm_list1 = nn.ModuleList(self.lstm_list1)
        self.lstm_list2 = [CLSTM(out_channels=4) for i in range(6)]                
        self.lstm_list2 = nn.ModuleList(self.lstm_list2)        
        self.lstm_list3 = [CLSTM(in_channels=4, out_channels=8) for i in range(6)]                
        self.lstm_list3 = nn.ModuleList(self.lstm_list3)
        self.lstm_list4 = [CLSTM(in_channels=4, out_channels=8) for i in range(6)]                
        self.lstm_list4 = nn.ModuleList(self.lstm_list4)        
        self.conv1 = nn.Conv3d(16, 64, kernel_size = 3, stride=4)
        self.conv2 = nn.Conv3d(64, 128, kernel_size = 3, stride=4)
        # self.conv3 = nn.Conv3d(128, 128, kernel_size = 3, stride=2)
        self.fc1 = nn.Linear(23040, 256)
        self.fc2 = nn.Linear(256, 6)
        ## for each direction, fist do convolution and then use the result for gates and memories

    def forward(self, x1, x2):        
        h_list1 = [self.lstm_list1[i](x1, i) for i in range(6)]
        out1 = torch.mean(torch.stack(h_list1), dim = 0)
        h_list2 = [self.lstm_list2[i](x2, i) for i in range(6)]
        out2 = torch.mean(torch.stack(h_list2), dim = 0)
        h_list3 = [self.lstm_list3[i](out1, i) for i in range(6)]
        out3 = torch.mean(torch.stack(h_list3), dim = 0)
        h_list4 = [self.lstm_list4[i](out2, i) for i in range(6)]
        out4 = torch.mean(torch.stack(h_list4), dim = 0)
        out = torch.cat((out3, out4), 1)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        # out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)        
        out = F.relu(self.fc1(out))        
        return self.fc2(out)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

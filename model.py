import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, in_dim, h_dim, edge_dim, part, task='regress'):
        super(Net, self).__init__()
        self.part = part
        self.task = task
        if part == 'R':
            self.lin0 = Linear(in_dim, h_dim)
        self.conv = NNConv(h_dim, h_dim, Linear(edge_dim, h_dim * h_dim))
        self.gru = GRU(h_dim, h_dim)
        if part == 'P':
            if task == 'regress':
                self.lin1 = Linear(h_dim, 1)
            else:
                self.lin1 = Linear(h_dim, 2)
        elif part == 'S':
            self.lin1 = Linear(h_dim, 2)

    def forward(self, data, repr=None):
        if self.part == 'R':
            out = F.relu(self.lin0(data.x))
        else:
            out = repr
        h = out.unsqueeze(0)
        for _ in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        if self.part == 'R':
            return out
        out = global_mean_pool(out, data.batch)
        out = self.lin1(out)
        if self.part == 'S':
            return F.log_softmax(out, dim=1)
        if self.task == 'regress':
            return out.view(-1)
        return F.log_softmax(out, dim=1)


class twostep_net(torch.nn.Module):
    def __init__(self, in_dim, h_dim, edge_dim, task='regress'):
        super(twostep_net, self).__init__()
        self.task = task
        self.lin0 = Linear(in_dim, h_dim)
        self.conv = NNConv(h_dim, h_dim, Linear(edge_dim, h_dim * h_dim))
        self.gru = GRU(h_dim, h_dim)
        if task == 'regress':
            self.lin1 = Linear(h_dim, 1)
        else:
            self.lin1 = Linear(h_dim, 2)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = global_mean_pool(out, data.batch)
        out = self.lin1(out)
        if self.task == 'regress':
            return out.view(-1)
        return F.log_softmax(out, dim=1)

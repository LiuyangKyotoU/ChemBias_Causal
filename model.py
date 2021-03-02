import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_mean_pool

class Net(torch.nn.Module):
    def __init__(self, in_dim, h_dim, part, use_gru=False):
        super(Net, self).__init__()
        self.part = part
        self.use_gru = use_gru
        if part == 'R':
            self.lin0 = Linear(in_dim, h_dim)
        self.conv = NNConv(h_dim, h_dim, Linear(4, h_dim * h_dim))
        if use_gru:
            self.gru = GRU(h_dim, h_dim)
        if part == 'P':
            # self.lin1 = Sequential(Linear(h_dim, h_dim // 2), ReLU(), Linear(h_dim // 2, 1))
            self.lin1 = Linear(h_dim, 1)
        elif part == 'S':
            # self.lin1 = Sequential(Linear(h_dim, h_dim // 2), ReLU(), Linear(h_dim // 2, 2))
            self.lin1 = Linear(h_dim, 2)

    def forward(self, data, repr=None):
        if self.part == 'R':
            out = F.relu(self.lin0(data.x))
        else:
            out = repr
        if self.use_gru:
            h = out.unsqueeze(0)
            for _ in range(3):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
        else:
            for _ in range(3):
                out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        if self.part == 'R':
            return out
        out = global_mean_pool(out, data.batch)
        out = self.lin1(out)
        if self.part == 'S':
            return F.log_softmax(out, dim=1)
        return out.view(-1)

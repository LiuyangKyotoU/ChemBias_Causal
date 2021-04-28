import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_mean_pool


class ConvLayer(torch.nn.Module):
    def __init__(self, h_dim, e_dim, times=3):
        super(ConvLayer, self).__init__()
        nn = Sequential(Linear(e_dim, h_dim), ReLU(), Linear(h_dim, h_dim * h_dim))
        self.conv = NNConv(h_dim, h_dim, nn, aggr='mean')
        self.gru = GRU(h_dim, h_dim)
        self.times = times

    def forward(self, batch, out):
        h = out.unsqueeze(0)
        for _ in range(self.times):
            m = F.relu(self.conv(out, batch.edge_index, batch.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return out


class BaselineRegressNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim, times):
        super(BaselineRegressNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))

    def forward(self, batch):
        out = self.lin1(global_mean_pool(self.conv_layer(batch, self.lin0(batch.x)), batch.batch))
        return out.view(-1)


class IpsClassifyNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim, times):
        super(IpsClassifyNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, batch):
        out = self.lin1(global_mean_pool(self.conv_layer(batch, self.lin0(batch.x)), batch.batch))
        return F.log_softmax(out, dim=1)


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DirlNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim, times):
        super(DirlNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.feature_conv_layer = ConvLayer(h_dim, e_dim, times)
        self.label_conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))
        self.domain_conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin2 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, batch, alpha):
        out = self.feature_conv_layer(batch, self.lin0(batch.x))
        r_out = ReverseLayerF.apply(out, alpha)
        label_out = self.lin1(global_mean_pool(self.label_conv_layer(batch, out), batch.batch))
        domain_out = self.lin2(global_mean_pool(self.domain_conv_layer(batch, r_out), batch.batch))
        return label_out.view(-1), F.log_softmax(domain_out, dim=1)


class CausalFeatureNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim, times):
        super(CausalFeatureNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim, times)

    def forward(self, batch):
        return self.conv_layer(batch, self.lin0(batch.x))


class CausalRegressNet(torch.nn.Module):
    def __init__(self, h_dim, e_dim, times):
        super(CausalRegressNet, self).__init__()
        self.conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))

    def forward(self, batch, out):
        out = self.lin1(global_mean_pool(self.conv_layer(batch, out), batch.batch))
        return out.view(-1)


class CausalClassifyNet(torch.nn.Module):
    def __init__(self, h_dim, e_dim, times):
        super(CausalClassifyNet, self).__init__()
        self.conv_layer = ConvLayer(h_dim, e_dim, times)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, batch, out):
        out = self.lin1(global_mean_pool(self.conv_layer(batch, out), batch.batch))
        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    from torch_geometric.datasets import QM9
    from torch_geometric.data import DataLoader

    dataset = QM9('data/QM9')
    loader = DataLoader(dataset, batch_size=6)
    data = iter(loader).next()

    model = BaselineRegressNet(11, 32, 4, 6)
    print(model(data))

    model = DirlNet(11, 32, 4, 3)
    print(model(data, 1))

    R = CausalFeatureNet(11, 32, 4, 3)
    D = CausalClassifyNet(32, 4, 3)
    L = CausalRegressNet(32, 4, 3)
    print(R(data))
    print(D(data, R(data)), L(data, R(data)))

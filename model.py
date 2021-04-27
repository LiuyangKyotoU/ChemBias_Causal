import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_mean_pool


class ConvLayer(torch.nn.Module):
    def __init__(self, h_dim, e_dim):
        super(ConvLayer, self).__init__()
        nn = Sequential(Linear(e_dim, h_dim), ReLU(), Linear(h_dim, h_dim * h_dim))
        self.conv = NNConv(h_dim, h_dim, nn, aggr='mean')
        self.gru = GRU(h_dim, h_dim)

    def forward(self, data, out):
        h = out.unsqueeze(0)
        for _ in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return out


class BaselineRegressNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim):
        super(BaselineRegressNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))

    def forward(self, data):
        out = self.lin1(global_mean_pool(self.conv_layer(data, self.lin0(data.x)), data.batch))
        return out.view(-1)


class IPSClassifyNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim):
        super(IPSClassifyNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, data):
        out = self.lin1(global_mean_pool(self.conv_layer(data, self.lin0(data.x)), data.batch))
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


class DIRLNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim):
        super(DIRLNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.feature_conv_layer = ConvLayer(h_dim, e_dim)
        self.label_conv_layer = ConvLayer(h_dim, e_dim)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))
        self.domain_conv_layer = ConvLayer(h_dim, e_dim)
        self.lin2 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, data, alpha):
        out = self.feature_conv_layer(data, self.lin0(data.x))
        r_out = ReverseLayerF.apply(out, alpha)
        label_out = self.lin1(global_mean_pool(out, data.batch))
        domain_out = self.lin2(global_mean_pool(r_out, data.batch))
        return label_out.view(-1), F.log_softmax(domain_out, dim=1)


class CausalFeatureNet(torch.nn.Module):
    def __init__(self, i_dim, h_dim, e_dim):
        super(CausalFeatureNet, self).__init__()
        self.lin0 = Sequential(Linear(i_dim, h_dim), ReLU())
        self.conv_layer = ConvLayer(h_dim, e_dim)

    def forward(self, data):
        return self.conv_layer(data, self.lin0(data.x))


class CausalRegressNet(torch.nn.Module):
    def __init__(self, h_dim, e_dim):
        super(CausalRegressNet, self).__init__()
        self.conv_layer = ConvLayer(h_dim, e_dim)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 1))

    def forward(self, data, out):
        out = self.lin1(global_mean_pool(self.conv_layer(data, out), data.batch))
        return out.view(-1)


class CausalClassifyNet(torch.nn.Module):
    def __init__(self, h_dim, e_dim):
        super(CausalClassifyNet, self).__init__()
        self.conv_layer = ConvLayer(h_dim, e_dim)
        self.lin1 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 2))

    def forward(self, data, out):
        out = self.lin1(global_mean_pool(self.conv_layer(data, out), data.batch))
        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    from torch_geometric.datasets import QM9
    from torch_geometric.data import DataLoader

    dataset = QM9('data/QM9')
    loader = DataLoader(dataset, batch_size=6)
    data = iter(loader).next()
    model = BaselineRegressNet(11, 32, 4)
    print(model(data))

    R = CausalFeatureNet(11, 32, 4)
    D = CausalClassifyNet(32, 4)
    L = CausalRegressNet(32, 4)
    print(R(data))
    print(D(data, R(data)), L(data, R(data)))

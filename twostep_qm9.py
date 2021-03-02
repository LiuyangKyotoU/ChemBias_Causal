import torch
import copy
import argparse
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_mean_pool

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, help='trial number')
parser.add_argument('--bias', type=str, help='101, 001, etc.')
args = parser.parse_args()

dataset = QM9('data/QM9')
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std

dic = torch.load('sampling/qm9/' + str(args.trial) + '.pt')
test_dataset = dataset[dic['test_index']]
train_dataset = dataset[dic['train_index'][args.bias]]
val_dataset = dataset[dic['val_index'][args.bias]]
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)


class net(torch.nn.Module):
    def __init__(self, mode='classify'):
        super(net, self).__init__()
        self.mode = mode
        self.lin0 = Linear(11, 32)
        self.conv = NNConv(32, 32, Linear(4, 32 * 32))
        self.gru = GRU(32, 32)
        if mode == 'classify':
            self.lin1 = Linear(32, 2)
        elif mode == 'regress':
            self.lin1 = Linear(32, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = global_mean_pool(out, data.batch)
        out = self.lin1(out)
        if self.mode == 'classify':
            return F.log_softmax(out, dim=1)
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = net(mode='classify').to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
for epoch in range(30):
    print(epoch)
    classifier.train()
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    for _ in range(len(train_iter)):
        train_data = train_iter.next().to(device)
        test_data = test_iter.next().to(device)
        optimizer.zero_grad()
        select_loss = F.nll_loss(torch.cat((classifier(train_data),
                                            classifier(test_data))),
                                 torch.cat((torch.ones(train_data.num_graphs),
                                            torch.zeros(test_data.num_graphs))).to(torch.int64).to(device))
        select_loss.backward()
        optimizer.step()
    scheduler.step()

path = 'qm9-twostep-trail({})-bias({})'.format(args.trial, args.bias)
mae_save = torch.zeros(12).to(torch.float32)
for target in range(12):
    print(target)
    regresser = net(mode='regress').to(device)
    optimizer = torch.optim.Adam(regresser.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
    best_val_mae = float('inf')
    best_model_dict = copy.deepcopy(regresser.state_dict())
    for epoch in range(200):
        regresser.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            weights = 1 / torch.exp(classifier(data)[:, 1])
            pred_loss = F.mse_loss(regresser(data), data.y[:, target], reduction='none')
            pred_loss = pred_loss * weights
            pred_loss = pred_loss.mean()
            pred_loss.backward()
            optimizer.step()
        scheduler.step()
        val_mae = 0
        for val_data in val_loader:
            val_data = val_data.to(device)
            val_mae += (regresser(val_data) * std[:, target].item() - val_data.y[:, target]).abs().sum().item()
        val_mae = val_mae / len(val_loader.dataset)
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_model_dict = copy.deepcopy(regresser.state_dict())
    regresser = net(mode='regress').to(device)
    regresser.load_state_dict(best_model_dict)
    test_mae = 0
    for test_data in test_loader:
        test_data = test_data.to(device)
        test_mae += (regresser(test_data) * std[:, target].item() - test_data.y[:, target]).abs().sum().item()
    test_mae = test_mae / len(test_loader.dataset)
    mae_save[target] = test_mae
torch.save(mae_save, 'results/' + path + '_mae.pt')

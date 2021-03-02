import torch
import copy
import argparse
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from geomloss import SamplesLoss
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, help='trial number')
parser.add_argument('--bias', type=str, help='101, 001, etc.')
parser.add_argument('--alpha', default=10, type=float, help='weight of IPM')
parser.add_argument('--beta', default=0.1, type=float, help='weight of Selection prediction loss. '
                                                            'If beta==0 means no weight for prediction loss')
args = parser.parse_args()

dataset = QM9('data/QM9')
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
# mean, std = mean[:, target].item(), std[:, target].item()

dic = torch.load('sampling/qm9/' + str(args.trial) + '.pt')
test_dataset = dataset[dic['test_index']]
train_dataset = dataset[dic['train_index'][args.bias]]
val_dataset = dataset[dic['val_index'][args.bias]]
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

path = 'qm9-trail({})-bias({})-alpha({})-beta({})'.format(args.trial, args.bias, args.alpha, args.beta)
writer = SummaryWriter(log_dir='runs/' + path)
model_save = {i: None for i in range(12)}
mae_save = torch.zeros(12).to(torch.float32)
for target in range(12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h_dim = 32
    R = Net(11, h_dim, 'R', use_gru=True).to(device)  # 不用gru的话会非常糟糕
    P = Net(None, h_dim, 'P', use_gru=True).to(device)
    S = Net(None, h_dim, 'S', use_gru=True).to(device)

    optimizer = torch.optim.Adam(list(R.parameters()) + list(P.parameters()) + list(S.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
    disc_loss_func = SamplesLoss('sinkhorn')

    best_val_mae = float('inf')
    best_model_dict = {'R': copy.deepcopy(R.state_dict()),
                       'P': copy.deepcopy(P.state_dict()),
                       'S': copy.deepcopy(S.state_dict())}
    for epoch in range(200):
        # print(epoch, scheduler.optimizer.param_groups[0]['lr'])
        R.train()
        P.train()
        S.train()
        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        train_loss = {'ipm': 0, 'select': 0, 'pred': 0, 'all': 0}
        for _ in range(len(train_iter)):
            train_data = train_iter.next().to(device)
            test_data = test_iter.next().to(device)
            optimizer.zero_grad()

            repr_train = R(train_data)
            repr_test = R(test_data)
            ipm_loss = disc_loss_func(global_mean_pool(repr_train, train_data.batch),
                                      global_mean_pool(repr_test, test_data.batch))

            prob_pred_train = S(train_data, repr_train)
            select_loss = F.nll_loss(torch.cat((prob_pred_train,
                                                S(test_data, repr_test))),
                                     torch.cat((torch.ones(train_data.num_graphs),
                                                torch.zeros(test_data.num_graphs))).to(torch.int64).to(device))

            weights = 1 / (2 * torch.exp(prob_pred_train[:, 1]))
            pred_loss = F.mse_loss(P(train_data, repr_train), train_data.y[:, target], reduction='none')
            if args.beta != 0:
                pred_loss = pred_loss * weights
            pred_loss = pred_loss.mean()

            loss = pred_loss + args.alpha * ipm_loss + args.beta * select_loss
            loss.backward()
            optimizer.step()

            train_loss['all'] += loss.item() * train_data.num_graphs
            train_loss['pred'] += pred_loss.item() * train_data.num_graphs
            train_loss['ipm'] += ipm_loss.item() * train_data.num_graphs
            train_loss['select'] += select_loss.item() * train_data.num_graphs
        scheduler.step()

        R.eval()
        P.eval()
        S.eval()
        val_mae = 0
        for val_data in val_loader:
            val_data = val_data.to(device)
            val_mae += (P(val_data, R(val_data)) * std[:, target].item() - val_data.y[:, target]).abs().sum().item()
        val_mae = val_mae / len(val_loader.dataset)
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_model_dict['R'] = copy.deepcopy(R.state_dict())
            best_model_dict['P'] = copy.deepcopy(P.state_dict())
            best_model_dict['S'] = copy.deepcopy(S.state_dict())

        writer.add_scalar('train/target' + str(target) + '/all',
                          train_loss['all'] / len(train_loader.dataset), epoch)
        writer.add_scalar('train/target' + str(target) + '/pred',
                          train_loss['pred'] / len(train_loader.dataset), epoch)
        writer.add_scalar('train/target' + str(target) + '/ipm',
                          args.alpha * train_loss['ipm'] / len(train_loader.dataset), epoch)
        writer.add_scalar('train/target' + str(target) + '/select',
                          args.beta * train_loss['select'] / len(train_loader.dataset), epoch)
        writer.add_scalar('val/target' + str(target), val_mae, epoch)

    R = Net(11, h_dim, 'R', use_gru=True).to(device)
    P = Net(None, h_dim, 'P', use_gru=True).to(device)
    S = Net(None, h_dim, 'S', use_gru=True).to(device)
    R.load_state_dict(best_model_dict['R'])
    P.load_state_dict(best_model_dict['P'])
    S.load_state_dict(best_model_dict['S'])
    test_mae = 0
    for test_data in test_loader:
        test_data = test_data.to(device)
        test_mae += (P(test_data, R(test_data)) * std[:, target].item() - test_data.y[:, target]).abs().sum().item()
    test_mae = test_mae / len(test_loader.dataset)

    model_save[target] = best_model_dict
    mae_save[target] = test_mae

torch.save(model_save, 'results/' + path + '_model.pt')
torch.save(mae_save, 'results/' + path + '_mae.pt')

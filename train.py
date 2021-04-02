import torch
import copy
import argparse
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.datasets import MoleculeNet
from geomloss import SamplesLoss
from model import Net, twostep_net
import math

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, help='qm9_0, qm9_1, ..., zinc, ...')
parser.add_argument('--trial', type=int, help='trial number')
args = parser.parse_args()


def regress1(path: 'task-scenario-trial',
             test_dataset, train_dataset, val_dataset, batch_size,
             in_dim, h_dim, edge_dim,
             std: float, eval='mae',
             epoch_num1=30, epoch_num2=200):
    print('{} Train Using BaseLine and Two-step Starts'.format(path))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = twostep_net(in_dim, h_dim, edge_dim, task='classify').to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    for epoch in range(epoch_num1):
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

    for method in ['baseline', 'twostep']:
        print('Method: {}'.format(method))
        regresser = twostep_net(in_dim, h_dim, edge_dim, task='regress').to(device)
        optimizer = torch.optim.Adam(regresser.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
        best_val_error = float('inf')
        best_model_dict = copy.deepcopy(regresser.state_dict())
        # classifier.eval()
        for epoch in range(epoch_num2):
            regresser.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                pred_loss = F.mse_loss(regresser(data), data.y, reduction='none')
                if method == 'twostep':
                    weights = 1 / torch.exp(classifier(data)[:, 1])
                    pred_loss = pred_loss * weights
                pred_loss = pred_loss.mean()
                pred_loss.backward()
                optimizer.step()
            scheduler.step()
            regresser.eval()
            val_error = 0
            for data in val_loader:
                data = data.to(device)
                if eval == 'mae':
                    val_error += (regresser(data) * std - data.y * std).abs().sum().item()
                elif eval == 'rmse':
                    val_error += ((regresser(data) * std - data.y * std) ** 2).sum().item()
            val_error = val_error / len(val_loader.dataset)
            if val_error < best_val_error:
                best_val_error = val_error
                best_model_dict = copy.deepcopy(regresser.state_dict())
            # print(val_error)
        regresser = twostep_net(in_dim, h_dim, edge_dim, task='regress').to(device)
        regresser.load_state_dict(best_model_dict)
        torch.save(best_model_dict, 'results/' + path + '-' + method + '.pt')
        regresser.eval()
        test_error = 0
        for data in test_loader:
            data = data.to(device)
            if eval == 'mae':
                test_error += (regresser(data) * std - data.y * std).abs().sum().item()
            elif eval == 'rmse':
                test_error += ((regresser(data) * std - data.y * std) ** 2).sum().item()
        test_error = test_error / len(test_loader.dataset)
        if eval == 'rmse':
            test_error = math.sqrt(test_error)
        with open('results/' + path + '.txt', 'a') as f:
            f.write(str(test_error) + '\t')
    print('{} Train Using BaseLine and Two-step Ends'.format(path))


def regress2(path: 'task-scenario-trial',
             test_dataset, train_dataset, val_dataset, batch_size,
             in_dim, h_dim, edge_dim,
             std: float, eval='mae',
             epoch_num=200):
    print('{} Train Using Causal Starts'.format(path))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for alpha in [0, 100]:
        for beta in [0, 0.1]:
            if alpha == beta:
                continue
            print('Params: alpha:{} beta:{}'.format(alpha, beta))
            R = Net(in_dim, h_dim, edge_dim, 'R').to(device)
            P = Net(None, h_dim, edge_dim, 'P').to(device)
            S = Net(None, h_dim, edge_dim, 'S').to(device)
            optimizer = torch.optim.Adam(list(R.parameters()) + list(P.parameters()) + list(S.parameters()), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
            disc_loss_func = SamplesLoss('sinkhorn')
            best_val_error = float('inf')
            best_model_dict = {'R': copy.deepcopy(R.state_dict()),
                               'P': copy.deepcopy(P.state_dict()),
                               'S': copy.deepcopy(S.state_dict())}
            for epoch in range(epoch_num):
                # print(epoch, scheduler.optimizer.param_groups[0]['lr'])
                R.train()
                P.train()
                S.train()
                train_iter = iter(train_loader)
                test_iter = iter(test_loader)
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
                    pred_loss = F.mse_loss(P(train_data, repr_train), train_data.y, reduction='none')
                    pred_loss = pred_loss * weights
                    pred_loss = pred_loss.mean()
                    loss = pred_loss + alpha * ipm_loss + beta * select_loss
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                R.eval()
                P.eval()
                S.eval()
                val_error = 0
                for data in val_loader:
                    data = data.to(device)
                    if eval == 'mae':
                        val_error += (P(data, R(data)) * std - data.y * std).abs().sum().item()
                    elif eval == 'rmse':
                        val_error += ((P(data, R(data)) * std - data.y * std) ** 2).sum().item()
                val_error = val_error / len(val_loader.dataset)
                if val_error <= best_val_error:
                    best_val_error = val_error
                    best_model_dict['R'] = copy.deepcopy(R.state_dict())
                    best_model_dict['P'] = copy.deepcopy(P.state_dict())
                    best_model_dict['S'] = copy.deepcopy(S.state_dict())
            R = Net(in_dim, h_dim, edge_dim, 'R').to(device)
            P = Net(None, h_dim, edge_dim, 'P').to(device)
            S = Net(None, h_dim, edge_dim, 'S').to(device)
            R.load_state_dict(best_model_dict['R'])
            P.load_state_dict(best_model_dict['P'])
            S.load_state_dict(best_model_dict['S'])
            torch.save(best_model_dict, 'results/' + path + '-' + str(alpha) + '_' + str(beta) + '.pt')
            R.eval()
            P.eval()
            S.eval()
            test_error = 0
            for data in test_loader:
                data = data.to(device)
                if eval == 'mae':
                    test_error += (P(data, R(data)) * std - data.y * std).abs().sum().item()
                elif eval == 'rmse':
                    test_error += ((P(data, R(data)) * std - data.y * std) ** 2).sum().item()
            test_error = test_error / len(test_loader.dataset)
            if eval == 'rmse':
                test_error = math.sqrt(test_error)
            with open('results/' + path + '.txt', 'a') as f:
                f.write(str(test_error) + '\t')
    print('{} Train Using Causal Ends'.format(path))


task = args.task
trial = args.trial

# QM9
if task[:3] == 'qm9':
    target = int(task[4:])


    class MyTransform(object):
        def __call__(self, data):
            data.y = data.y[:, target]
            return data


    scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']
    dataset = QM9('data/QM9', transform=MyTransform())
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()
    for s in scenarios:
        path = '{}-{}-{}'.format(task, s, trial)
        dic = torch.load('sampling/qm9/' + str(trial) + '.pt')
        test_dataset = dataset[dic['test_index']]
        train_dataset = dataset[dic['train_index'][s]]
        val_dataset = dataset[dic['val_index'][s]]
        regress1(path,
                 test_dataset, train_dataset, val_dataset, 256,
                 11, 32, 4,
                 std, 'mae',
                 30, 200)
        regress2(path,
                 test_dataset, train_dataset, val_dataset, 256,
                 11, 32, 4,
                 std, 'mae',
                 200)

# ZINC
if task == 'zinc':
    class MyTransform(object):
        def __call__(self, data):
            data.x = F.one_hot(data.x.view(-1), num_classes=28).to(torch.float32)
            data.edge_attr = F.one_hot(data.edge_attr, num_classes=4).to(torch.float32)
            return data


    scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']
    dataset = ZINC('data/ZINC', transform=MyTransform())
    mean = dataset.data.y.mean()
    std = dataset.data.y.std()
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.item(), std.item()
    for s in scenarios:
        path = '{}-{}-{}'.format(task, s, trial)
        dic = torch.load('sampling/zinc/' + str(trial) + '.pt')
        test_dataset = dataset[dic['test_index']]
        train_dataset = dataset[dic['train_index'][s]]
        val_dataset = dataset[dic['val_index'][s]]
        regress1(path,
                 test_dataset, train_dataset, val_dataset, 256,
                 28, 32, 4,
                 std, 'mae',
                 30, 200)
        regress2(path,
                 test_dataset, train_dataset, val_dataset, 256,
                 28, 32, 4,
                 std, 'mae',
                 200)

# ESOL
if task == 'esol':
    class MyTransform(object):
        def __call__(self, data):
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.y = data.y[:, 0]
            return data


    scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']
    dataset = MoleculeNet('data/MolNet', 'ESOL', transform=MyTransform())
    mean = dataset.data.y.mean()
    std = dataset.data.y.std()
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.item(), std.item()
    for s in scenarios:
        path = '{}-{}-{}'.format(task, s, trial)
        dic = torch.load('sampling/esol/' + str(trial) + '.pt')
        test_dataset = dataset[dic['test_index']]
        train_dataset = dataset[dic['train_index'][s]]
        val_dataset = dataset[dic['val_index'][s]]
        regress1(path,
                 test_dataset, train_dataset, val_dataset, 8,
                 9, 32, 3,
                 std, 'rmse',
                 30, 200)
        regress2(path,
                 test_dataset, train_dataset, val_dataset, 8,
                 9, 32, 3,
                 std, 'rmse',
                 200)


# LIPO
if task == 'lipo':
    class MyTransform(object):
        def __call__(self, data):
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.y = data.y[:, 0]
            return data


    scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']
    dataset = MoleculeNet('data/MolNet', 'LIPO', transform=MyTransform())
    mean = dataset.data.y.mean()
    std = dataset.data.y.std()
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.item(), std.item()
    for s in scenarios:
        path = '{}-{}-{}'.format(task, s, trial)
        dic = torch.load('sampling/lipo/' + str(trial) + '.pt')
        test_dataset = dataset[dic['test_index']]
        train_dataset = dataset[dic['train_index'][s]]
        val_dataset = dataset[dic['val_index'][s]]
        regress1(path,
                 test_dataset, train_dataset, val_dataset, 16,
                 9, 32, 3,
                 std, 'rmse',
                 30, 200)
        regress2(path,
                 test_dataset, train_dataset, val_dataset, 16,
                 9, 32, 3,
                 std, 'rmse',
                 200)


# FREESOLV
if task == 'freesolv':
    class MyTransform(object):
        def __call__(self, data):
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.y = data.y[:, 0]
            return data


    scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']
    dataset = MoleculeNet('data/MolNet', 'FREESOLV', transform=MyTransform())
    mean = dataset.data.y.mean()
    std = dataset.data.y.std()
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.item(), std.item()
    for s in scenarios:
        path = '{}-{}-{}'.format(task, s, trial)
        dic = torch.load('sampling/freesolv/' + str(trial) + '.pt')
        test_dataset = dataset[dic['test_index']]
        train_dataset = dataset[dic['train_index'][s]]
        val_dataset = dataset[dic['val_index'][s]]
        regress1(path,
                 test_dataset, train_dataset, val_dataset, 2,
                 9, 32, 3,
                 std, 'rmse',
                 30, 200)
        regress2(path,
                 test_dataset, train_dataset, val_dataset, 2,
                 9, 32, 3,
                 std, 'rmse',
                 200)

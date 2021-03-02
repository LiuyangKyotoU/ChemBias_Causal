import torch
import argparse
from torch_geometric.datasets import QM9
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, help='trial number')
args = parser.parse_args()

# bias_scenarios = ['0000', '1000', '0100', '0010', '0001',
#                   '1100', '1010', '1001', '0110', '0101', '0011',
#                   '1110', '1101', '1011', '0111',
#                   '1111']
bias_scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']


def size_p(data):
    return 1 / (1 + np.exp(0.5 * (data.x.shape[0] - 18)))


def prop_p(data):
    return 1 / (1 + np.exp(-50 * ((data.edge_attr[:, 0].sum() / data.edge_attr.shape[0]).item() - 0.92)))


def gap_p(data):
    return 1 / (1 + np.exp(-2 * (data.y[0, 4].item() - 6.9)))


# def u298_p(data):
#     return 1 / (1 + np.exp(-0.001 * (data.y[0, 8].item() + 11182)))


def bias_sampler(dataset, pattern, index, size):
    tmp = []
    for i in index.tolist():
        p = 0
        data = dataset[i]
        if pattern[0] == '1':
            p += size_p(data)
        if pattern[1] == '1':
            p += prop_p(data)
        if pattern[2] == '1':
            p += gap_p(data)
        # if pattern[3] == '1':
        #     p += u298_p(data)
        if pattern.count('1') == 0:
            p = 0.5
        else:
            p /= pattern.count('1')
        tmp.append(p)
    tmp = np.array(tmp)
    tmp = tmp / tmp.sum()
    ans = np.random.choice(index, size, replace=False, p=tmp)
    return torch.Tensor(ans).to(torch.int64)


def sampler():
    dataset = QM9('data/QM9')
    n = len(dataset)
    dic = {
        'test_index': None,
        'train_index': {i: None for i in bias_scenarios},
        'val_index': {i: None for i in bias_scenarios}
    }
    index = torch.randperm(n)
    dic['test_index'] = index[:n // 10]
    for pattern in bias_scenarios:
        print(args.trial, pattern)
        tmp = bias_sampler(dataset, pattern, index[n // 10:], n // 7)  # train:val = 7:3
        tmp = tmp[torch.randperm(n // 7)]
        dic['train_index'][pattern] = tmp[:n // 10]
        dic['val_index'][pattern] = tmp[n // 10:]

    torch.save(dic, 'sampling/qm9/' + str(args.trial) + '.pt')


sampler()

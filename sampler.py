import torch
from torch_geometric.datasets import QM9, ZINC
import numpy as np

scenarios = ['000', '100', '010', '001', '110', '101', '011', '111']


def sigmoid(x, a, b):
    return 1 / (1 + torch.exp(-a * (x - b)))


def bias_sampler(arrs, params, s, index, size):
    assert len(arrs) == len(params) == len(s)
    if '1' not in s:
        n = len(index)
        return torch.tensor(np.random.choice(index, size, replace=False, p=[1 / n] * n))
    tmp = torch.tensor([0] * len(arrs[0])).to(torch.float32)
    for i in range(len(s)):
        if s[i] == '0':
            continue
        tmp += sigmoid(arrs[i], params[i][0], params[i][1])
    tmp = tmp / tmp.sum()
    return torch.tensor(np.random.choice(index, size, replace=False, p=tmp.numpy()))


def zinc_bias():
    zinc = ZINC('data/ZINC')
    n = len(zinc)
    # molecular size, molecular size, molecular gap value
    ind1, ind2, ind3 = [], [], []
    for data in zinc:
        ind1.append(data.x.shape[0])
        ind2.append(torch.nonzero(data.edge_attr == 1).shape[0] / data.edge_attr.shape[0])
        ind3.append(data.y.item())
    ind1 = torch.tensor(ind1).to(torch.float32)
    ind2 = torch.tensor(ind2).to(torch.float32)
    ind3 = torch.tensor(ind3).to(torch.float32)
    for trial in range(10):
        index = torch.randperm(n)
        test_index = index[:n // 20]
        other_index = index[n // 20:]
        dic = {'test_index': test_index, 'train_index': {}, 'val_index': {}}
        for s in scenarios:
            biased = bias_sampler([ind1[other_index], ind2[other_index], ind3[other_index]],
                                  [[-2, 20], [-50, 0.7], [-2, -2]],
                                  s, other_index, n // 14)
            biased = biased[torch.randperm(biased.shape[0])]
            train_index = biased[:n // 20]
            val_index = biased[n // 20:]
            dic['train_index'][s] = train_index
            dic['val_index'][s] = val_index
        torch.save(dic, 'sampling/zinc/' + str(trial) + '.pt')


def qm9_bias():
    qm9 = QM9('data/QM9')
    n = len(qm9)
    # molecular size, molecular size, molecular gap value
    ind1, ind2, ind3 = [], [], []
    for data in qm9:
        ind1.append(data.x.shape[0])
        ind2.append((data.edge_attr[:, 0].sum() / data.edge_attr.shape[0]).item())
        ind3.append(data.y[0, 4].item())
    ind1 = torch.tensor(ind1).to(torch.float32)
    ind2 = torch.tensor(ind2).to(torch.float32)
    ind3 = torch.tensor(ind3).to(torch.float32)

    for trial in range(10):
        index = torch.randperm(n)
        test_index = index[:n // 10]
        other_index = index[n // 10:]
        dic = {'test_index': test_index, 'train_index': {}, 'val_index': {}}
        for s in scenarios:
            biased = bias_sampler([ind1[other_index], ind2[other_index], ind3[other_index]],
                                  [[-2, 13], [-50, 0.8], [2, 9]],
                                  s, other_index, n // 7)
            biased = biased[torch.randperm(biased.shape[0])]
            train_index = biased[:n // 10]
            val_index = biased[n // 10:]
            dic['train_index'][s] = train_index
            dic['val_index'][s] = val_index
        torch.save(dic, 'sampling/qm9/' + str(trial) + '.pt')


def curve(arr, a, b, x):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.hist(arr.numpy(), bins=100)
    ax2 = ax1.twinx()
    y = sigmoid(x, a, b).numpy()
    ax2.plot(x, y, color='red')
    plt.show()


def diff_dis_qm9(qm9, biased, unbiased):
    import matplotlib.pyplot as plt
    s1, s2 = [], []
    p1, p2 = [], []
    y1, y2 = [], []
    for d in qm9[biased]:
        s1.append(d.x.shape[0])
        p1.append((d.edge_attr[:, 0].sum() / d.edge_attr.shape[0]).item())
        y1.append(d.y[0, 4].item())
    for d in qm9[unbiased]:
        s2.append(d.x.shape[0])
        p2.append((d.edge_attr[:, 0].sum() / d.edge_attr.shape[0]).item())
        y2.append(d.y[0, 4].item())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(s1, bins=100, alpha=0.5)
    ax1.hist(s2, bins=100, alpha=0.5)
    ax2.hist(p1, bins=100, alpha=0.5)
    ax2.hist(p2, bins=100, alpha=0.5)
    ax3.hist(y1, bins=100, alpha=0.5)
    ax3.hist(y2, bins=100, alpha=0.5)


def diff_dis_zinc(zinc, biased, unbiased):
    import matplotlib.pyplot as plt
    s1, s2 = [], []
    p1, p2 = [], []
    y1, y2 = [], []
    for d in zinc[biased]:
        s1.append(d.x.shape[0])
        p1.append(torch.nonzero(d.edge_attr == 1).shape[0] / d.edge_attr.shape[0])
        y1.append(d.y.item())
    for d in zinc[unbiased]:
        s2.append(d.x.shape[0])
        p2.append(torch.nonzero(d.edge_attr == 1).shape[0] / d.edge_attr.shape[0])
        y2.append(d.y.item())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(s1, bins=100, alpha=0.5)
    ax1.hist(s2, bins=100, alpha=0.5)
    ax2.hist(p1, bins=100, alpha=0.5)
    ax2.hist(p2, bins=100, alpha=0.5)
    ax3.hist(y1, bins=100, alpha=0.5)
    ax3.hist(y2, bins=100, alpha=0.5)
    ax3.set_xlim([-10, 5])


if __name__ == '__main__':
    # qm9 = QM9('data/QM9')
    # qm9_bias()
    # dic = torch.load('sampling/qm9/0.pt')
    # diff_dis_qm9(qm9, dic['train_index']['100'], dic['test_index'])

    zinc = ZINC('data/ZINC')
    zinc_bias()
    dic = torch.load('sampling/zinc/0.pt')
    diff_dis_zinc(zinc, dic['train_index']['100'], dic['test_index'])

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9, ZINC, MoleculeNet
import itertools


class Sampler(object):
    def __init__(self, dataset, f1_alpha, f2_alpha, f3_alpha):
        self.n = len(dataset)
        self.f1_tensor = self._get_mols_f1(dataset)
        self.f2_tensor = self._get_mols_f2(dataset)
        self.f3_tensor = self._get_mols_f3(dataset)
        self.f1_alpha = f1_alpha
        self.f2_alpha = f2_alpha
        self.f3_alpha = f3_alpha
        self.name = self.__class__.__name__ + ':'

    def _get_mols_f1(self, dataset):
        raise NotImplementedError

    def _get_mols_f2(self, dataset):
        raise NotImplementedError

    def _get_mols_f3(self, dataset):
        raise NotImplementedError

    def _sigmoid(self, x, a, b):
        return 1 / (1 + torch.exp(-a * (x - b)))

    def _save(self, test_ids, train_ids, val_ids, factors, trial):
        dic = {'test_ids': test_ids, 'train_ids': train_ids, 'val_ids': val_ids}
        torch.save(dic, 'sampling/' + self.name + '+'.join(factors) + '_' + str(trial) + '.pt')

    def run_all_sampling(self):
        factors = ['f1', 'f2', 'f3']
        for i in range(len(factors) + 1):
            for subset in itertools.combinations(factors, i):
                for trial in range(10):
                    self.sampling(subset, trial)

    def sampling(self, factors, trial):
        ids = torch.randperm(self.n)
        test_ids = ids[:self.n // 10]
        other_ids = ids[self.n // 10:]
        scores = torch.zeros(other_ids.shape[0]).to(torch.float32)
        if not factors:
            scores = torch.ones(other_ids.shape[0]).to(torch.float32)
        for factor in factors:
            t = self.__getattribute__(factor + '_tensor')
            a = self.__getattribute__(factor + '_alpha')
            scores += self._sigmoid(t[other_ids], a, t.median())
        scores = scores / scores.sum()
        train_val_ids = torch.tensor(
            np.random.choice(other_ids, self.n // 10 * 10 // 7, replace=False, p=scores.numpy()))
        train_val_ids = train_val_ids[torch.randperm(train_val_ids.shape[0])]
        train_ids = train_val_ids[:self.n // 10]
        val_ids = train_val_ids[self.n // 10:]
        self._save(test_ids, train_ids, val_ids, factors, trial)

    def draw(self, factor, bins, xlim=None):
        dic = torch.load('sampling/' + self.name + factor + '.pt')
        unbias_ids = dic['test_ids']
        bias_ids = dic['train_ids']
        t = self.__getattribute__(factor + '_tensor')
        a = self.__getattribute__(factor + '_alpha')
        tmp1 = t[unbias_ids].numpy()
        tmp2 = t[bias_ids].numpy()
        fig, ax = plt.subplots()
        ax.hist(tmp1, weights=np.ones_like(tmp1) / len(tmp1), bins=bins, alpha=0.5)
        ax.hist(tmp2, weights=np.ones_like(tmp2) / len(tmp2), bins=bins, alpha=0.5)
        ax_ = ax.twinx()
        x = torch.linspace(t.min(), t.max(), 100)
        ax_.plot(x.numpy(), self._sigmoid(x, a, t.median()).numpy())
        if xlim:
            ax.set_xlim([*xlim])
        plt.show()


class QM9Sampler(Sampler):
    def __init__(self, f1_alpha, f2_alpha, f3_alpha):
        dataset = QM9('data/QM9')
        super(QM9Sampler, self).__init__(dataset, f1_alpha, f2_alpha, f3_alpha)

    def _get_mols_f1(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.x.shape[0])
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_f2(self, dataset):
        ans = []
        for data in dataset:
            ans.append((data.edge_attr[:, 0].sum() / data.edge_attr.shape[0]).item())
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_f3(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.y[0, 4].item())
        return torch.tensor(ans).to(torch.float32)


if __name__ == '__main__':
    sampler = QM9Sampler(-1, -50, 2)
    sampler.run_all_sampling()
    # sampler.sampling('f1')
    # sampler.sampling('f2')
    # sampler.sampling('f3')
    # sampler.draw('f1', 40)
    # sampler.draw('f2', 20, [0.75, 1])
    # sampler.draw('f3', 40,[2,12])

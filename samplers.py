import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


class Sampler(object):
    def _get_mols_size(self, dataset):
        raise NotImplementedError

    def _get_mols_edge_prop(self, dataset):
        raise NotImplementedError

    def _get_mols_one_value(self, dataset):
        raise NotImplementedError

    def _get_one_factor_mean(self, dataset, factor):
        factor_func = self.__getattribute__('_get_mols_' + factor)
        return factor_func(dataset).mean()

    def _sigmoid(self, x, a, b):
        return 1 / (1 + torch.exp(-a * (x - b)))

    def _gaussian(self, x, mu, alpha):
        pass

    def _strategy_to_str(self, **kw):
        factors = kw.get('factors', {})
        tmp = ''
        for factor in factors:
            tmp += '-'.join([factor, *list(map(str, factors[factor]))])
        if not tmp:
            tmp = 'None'
        return '_'.join([kw.get('name'),
                         str(kw.get('test_train_frac', 0.1)),
                         kw.get('chance_type', 'sigmoid'),
                         tmp,
                         str(kw.get('trial'))])

    def _save(self, test_ids, train_ids, val_ids, **kw):
        dic = {'test_ids': test_ids, 'train_ids': train_ids, 'val_ids': val_ids}
        torch.save(dic, 'sampling/' + self._strategy_to_str(**kw) + '.pt')

    def sampling(self, dataset, **kw):
        frac = kw.get('test_train_frac', 0.1)
        factors = kw.get('factors', {})
        chance_type = kw.get('chance_type', 'sigmoid')
        chance_func = self.__getattribute__('_' + chance_type)

        n = len(dataset)
        ids = torch.randperm(n)
        test_ids = ids[:int(n * frac)]
        other_ids = ids[int(n * frac):]
        other_dataset = dataset[other_ids]

        scores = torch.zeros(len(other_dataset)).to(torch.float32)
        if not factors:
            scores = torch.ones(len(other_dataset)).to(torch.float32)
        for factor in factors:
            factor_func = self.__getattribute__('_get_mols_' + factor)
            scores += chance_func(factor_func(other_dataset), *factors[factor])
        scores = scores / scores.sum()

        train_val_ids = torch.tensor(
            np.random.choice(other_ids, int(10 / 7 * n * frac), replace=False, p=scores.numpy()))
        train_val_ids = train_val_ids[torch.randperm(train_val_ids.shape[0])]
        train_ids = train_val_ids[:int(n * frac)]
        val_ids = train_val_ids[int(n * frac):]
        self._save(test_ids, train_ids, val_ids, **kw)

    def draw(self, dataset, factor, bins, *args, **kw):
        dic = torch.load('sampling/' + self._strategy_to_str(**kw) + '.pt')

        unbiased_ids = dic['test_ids']
        biased_ids = dic['train_ids']
        unbiased_dataset = dataset[unbiased_ids]
        biased_dataset = dataset[biased_ids]

        fig, ax = plt.subplots()
        factor_func = self.__getattribute__('_get_mols_' + factor)
        tmp1 = factor_func(unbiased_dataset).numpy()
        tmp2 = factor_func(biased_dataset).numpy()
        ax.hist(tmp1, weights=np.ones_like(tmp1) / len(tmp1), bins=bins, alpha=0.5)
        ax.hist(tmp2, weights=np.ones_like(tmp2) / len(tmp2), bins=bins, alpha=0.5)
        if args:
            ax.set_xlim([args[0], args[1]])
        plt.show()


class QM9Sampler(Sampler):
    def _get_mols_size(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.x.shape[0])
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_edge_prop(self, dataset):
        ans = []
        for data in dataset:
            ans.append((data.edge_attr[:, 0].sum() / data.edge_attr.shape[0]).item())
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_one_value(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.y[0, 4].item())
        return torch.tensor(ans).to(torch.float32)


class ZINCSampler(Sampler):
    def _get_mols_size(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.x.shape[0])
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_edge_prop(self, dataset):
        ans = []
        for data in dataset:
            ans.append(torch.nonzero(data.edge_attr == 1).shape[0] / data.edge_attr.shape[0])
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_one_value(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.y.item())
        return torch.tensor(ans).to(torch.float32)


class MoleNetSampler(Sampler):
    def _get_mols_size(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.x.shape[0])
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_edge_prop(self, dataset):
        ans = []
        for data in dataset:
            ans.append((torch.nonzero(data.edge_attr[:, 0] == 1).shape[0] + 1) / (data.edge_attr.shape[0] + 1))
        return torch.tensor(ans).to(torch.float32)

    def _get_mols_one_value(self, dataset):
        ans = []
        for data in dataset:
            ans.append(data.y[0, 0].item())
        return torch.tensor(ans).to(torch.float32)


if __name__ == '__main__':
    from torch_geometric.datasets import QM9, ZINC

    dataset = QM9('data/QM9')
    sampler = QM9Sampler()
    params = {'size': [-1, 13], 'edge_prop': [-30, 0.7], 'one_value': [1.5, 9]}
    for i in range(len(params) + 1):
        for subset in itertools.combinations(params, i):
            for trial in range(10):
                strategy = {
                    'name': 'qm9',
                    'test_train_frac': 0.1,
                    'chance_type': 'sigmoid',
                    'factors': {f: params[f] for f in subset},
                    'trial': trial
                }
                sampler.sampling(dataset, **strategy)

    bins = {'size': 20, 'edge_prop': 20, 'one_value': 40}
    factor = 'edge_prop'
    strategy = {
        'name': 'qm9',
        'test_train_frac': 0.1,
        'chance_type': 'sigmoid',
        'factors': {factor: params[factor]},
        'trial': 0
    }
    sampler.draw(dataset, factor, bins[factor], 0.6, 1, **strategy)

    from torch_geometric.datasets import MoleculeNet
    #

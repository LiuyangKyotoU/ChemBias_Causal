import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.datasets import MoleculeNet


class Preprocessor:

    def _qm9(self, target):
        dataset = QM9('data/QM9', transform=QM9Transformer(target))
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        return dataset, std[:, target].item(), 11, 4

    def _zinc(self):
        dataset = ZINC('data/ZINC', transform=ZINCTransformer())
        mean = dataset.data.y.mean()
        std = dataset.data.y.std()
        dataset.data.y = (dataset.data.y - mean) / std
        return dataset, std.item(), 28, 4

    def _molenet(self, task):
        dataset = MoleculeNet('data/MolNet', task, transform=MolNetTransformer())
        mean = dataset.data.y.mean()
        std = dataset.data.y.std()
        dataset.data.y = (dataset.data.y - mean) / std
        return dataset, std.item(), 9, 3

    def _split(self, dataset, scenario):
        dic = torch.load('sampling/' + scenario + '.pt')
        return dataset[dic['test_ids']], dataset[dic['train_ids']], dataset[dic['val_ids']]

    def get_dataset(self, task, scenario):
        if task[:3] == 'qm9':
            target = int(task.split('_')[1])
            dataset, std, i_dim, e_dim = self._qm9(target)
        elif task == 'zinc':
            dataset, std, i_dim, e_dim = self._zinc()
        elif task in ['esol', 'lipo', 'freesolv']:
            dataset, std, i_dim, e_dim = self._molenet(task)

        test_dataset, train_dataset, val_dataset = self._split(dataset, scenario)
        return (test_dataset, train_dataset, val_dataset), std, i_dim, e_dim


class QM9Transformer(object):
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        data.y = data.y[:, self.target]
        return data


class ZINCTransformer(object):
    def __call__(self, data):
        data.x = F.one_hot(data.x.view(-1), num_classes=28).to(torch.float32)
        data.edge_attr = F.one_hot(data.edge_attr, num_classes=4).to(torch.float32)
        return data


class MolNetTransformer(object):
    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.y = data.y[:, 0]
        return data

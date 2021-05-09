import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from geomloss import SamplesLoss

import models
from preprocessor import Preprocessor
from evaluator import Evaluator


class Trainer:
    def __init__(self, task, scenario, **kw):
        self.task = task
        self.scenario = scenario
        self.name = task + '==>' + scenario

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_dim = kw.get('h_dim')
        self.times = kw.get('times')
        self.batch_size = kw.get('batch_size')
        self.lr = kw.get('lr')
        self.epoch = kw.get('epoch')

        self.datasets, self.std, self.i_dim, self.e_dim = Preprocessor().get_dataset(task, scenario)
        self.test_loader, self.train_loader, self.val_loader = self._create_loaders()
        self.batch_error_func, self.all_error_func = Evaluator().get_error_func(task)

    def _train(self, *args):
        raise NotImplementedError

    def _test(self, *args):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def _save(self, test_error, model_state_dic):
        with open('results/test_errors.txt', 'a') as f:
            f.write(self.name + '\t' + str(test_error) + '\n')
        torch.save(model_state_dic, 'results/' + self.name + '.pt')
        print('Result of {} saved!'.format(self.name))

    def _create_loaders(self, ):
        test_dataset, train_dataset, val_dataset = self.datasets
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        print('Load {} successfully!'.format(self.name))
        return test_loader, train_loader, val_loader


class BaselineTrainer(Trainer):
    def __init__(self, task, scenario, **kw):
        super(BaselineTrainer, self).__init__(task, scenario, **kw)
        self.name = 'Baseline' + str(self.times) + '==>' + self.name

    def _train(self, model, optimizer):
        model.train()
        loss_all = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch), batch.y)
            loss.backward()
            loss_all += loss.item() * batch.num_graphs
            optimizer.step()
        return loss_all / len(self.train_loader.dataset)

    def _test(self, model, loader):
        model.eval()
        error = 0
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = model(batch)
            error += self.batch_error_func(pred, batch.y, self.std)
        return self.all_error_func(error / len(loader.dataset))

    def run(self):
        model = models.BaselineRegressNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                               min_lr=0.00001)
        best_val_error = float('inf')
        best_model_dic = None
        for e in range(self.epoch):
            train_loss = self._train(model, optimizer)
            val_error = self._test(model, self.val_loader)
            scheduler.step(val_error)
            if val_error <= best_val_error:
                best_val_error = val_error
                best_model_dic = copy.deepcopy(model.state_dict())
            print(e, train_loss, val_error)
        model = models.BaselineRegressNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        model.load_state_dict(best_model_dic)
        test_error = self._test(model, self.test_loader)
        self._save(test_error, best_model_dic)


class IpsTrainer(Trainer):
    def __init__(self, task, scenario, **kw):
        super(IpsTrainer, self).__init__(task, scenario, **kw)
        self.first_train_epoch = kw.get('first_train_epoch')
        assert self.first_train_epoch is not None

        self.name = 'IPS' + str(self.times) + '==>' + self.name

    def _train(self, model, optimizer, classifier):
        classifier.eval()
        model.train()
        loss_all = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch), batch.y, reduction='none')
            with torch.no_grad():
                weights = 1 / (torch.exp(classifier(batch)[:, 1]) * 2)
            loss = loss * weights
            loss = loss.mean()
            loss.backward()
            loss_all += loss.item() * batch.num_graphs
            optimizer.step()
        return loss_all / len(self.train_loader.dataset)

    def _test(self, model, loader):
        model.eval()
        error = 0
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = model(batch)
            error += self.batch_error_func(pred, batch.y, self.std)
        return self.all_error_func(error / len(loader.dataset))

    def _first_train(self, model, optimizer, scheduler):
        unbias_dataset, bias_dataset, _ = self.datasets
        n = len(bias_dataset)
        val_size = int(n * 0.3)
        ids = torch.randperm(n)
        val_bias_dataset = bias_dataset[ids[:val_size]]
        val_unbias_dataset = unbias_dataset[ids[:val_size]]
        train_bias_dataset = bias_dataset[ids[val_size:]]
        train_unbias_dataset = unbias_dataset[ids[val_size:]]
        val_bias_loader = DataLoader(val_bias_dataset, batch_size=self.batch_size, shuffle=False)
        val_unbias_loader = DataLoader(val_unbias_dataset, batch_size=self.batch_size, shuffle=False)
        train_bias_loader = DataLoader(train_bias_dataset, batch_size=self.batch_size, shuffle=True)
        train_unbias_loader = DataLoader(train_unbias_dataset, batch_size=self.batch_size, shuffle=True)
        best_val_acc = -float('inf')
        best_model_dict = None
        for e in range(self.first_train_epoch):
            model.train()
            train_bias_iter = iter(train_bias_loader)
            train_unbias_iter = iter(train_unbias_loader)
            loss_all = 0
            for _ in range(len(train_bias_iter)):
                bias_batch = train_bias_iter.next().to(self.device)
                unbias_batch = train_unbias_iter.next().to(self.device)
                optimizer.zero_grad()
                loss = F.nll_loss(
                    torch.cat((model(bias_batch), model(unbias_batch))),
                    torch.cat((torch.ones(bias_batch.num_graphs),
                               torch.zeros(unbias_batch.num_graphs))).to(torch.int64).to(self.device)
                )
                loss.backward()
                loss_all += loss.item() * (bias_batch.num_graphs + unbias_batch.num_graphs)
                optimizer.step()
            loss_all = loss_all / (len(train_bias_loader.dataset) + len(train_unbias_loader.dataset))
            model.eval()
            correct = 0
            val_bias_iter = iter(val_bias_loader)
            val_unbias_iter = iter(val_unbias_loader)
            for _ in range(len(val_bias_iter)):
                bias_batch = val_bias_iter.next().to(self.device)
                unbias_batch = val_unbias_iter.next().to(self.device)
                with torch.no_grad():
                    pred = torch.cat((model(bias_batch), model(unbias_batch))).max(1)[1]
                correct += pred.eq(
                    torch.cat((torch.ones(bias_batch.num_graphs),
                               torch.zeros(unbias_batch.num_graphs))).to(torch.int64).to(self.device)
                ).sum().item()
            val_acc = correct / (len(val_bias_loader.dataset) + len(val_unbias_loader.dataset))
            scheduler.step()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_dict = copy.deepcopy(model.state_dict())
            print(e, loss_all, val_acc)
        return best_model_dict

    def run(self):
        # first step
        classifier = models.IpsClassifyNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.first_train_epoch // 2, gamma=0.1)
        classifier_best_stat_dic = self._first_train(classifier, optimizer, scheduler)
        classifier = models.IpsClassifyNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        classifier.load_state_dict(classifier_best_stat_dic)
        # second step
        model = models.BaselineRegressNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                               min_lr=0.00001)
        best_val_error = float('inf')
        best_model_dic = None
        for e in range(self.epoch):
            train_loss = self._train(model, optimizer, classifier)
            val_error = self._test(model, self.val_loader)
            scheduler.step(val_error)
            if val_error <= best_val_error:
                best_val_error = val_error
                best_model_dic = copy.deepcopy(model.state_dict())
            print(e, train_loss, val_error)
        model = models.BaselineRegressNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        model.load_state_dict(best_model_dic)
        test_error = self._test(model, self.test_loader)
        self._save(test_error, best_model_dic)


class DirlTrainer(Trainer):
    def __init__(self, task, scenario, **kw):
        super(DirlTrainer, self).__init__(task, scenario, **kw)
        self.name = 'Dirl' + '==>' + self.name

    def _train(self, model, optimizer, e):
        model.train()
        source_iter = iter(self.train_loader)  # bias_iter
        target_iter = iter(self.test_loader)  # unbias_iter
        loss_all = 0
        for i in range(len(source_iter)):
            p = (i + e * len(source_iter)) / self.epoch / len(source_iter)
            alpha = 2 / (1 + np.exp(-10 * p)) - 1

            optimizer.zero_grad()
            # {source / bias / train} domain
            batch = source_iter.next().to(self.device)
            label_out, domain_out = model(batch, alpha)
            loss = F.mse_loss(label_out, batch.y)
            loss += F.nll_loss(domain_out, torch.zeros(batch.num_graphs).to(torch.int64).to(self.device))
            # {target / unbias / test} domain
            batch = target_iter.next().to(self.device)
            _, domain_out = model(batch, alpha)
            loss += F.nll_loss(domain_out, torch.ones(batch.num_graphs).to(torch.int64).to(self.device))
            loss.backward()
            loss_all += loss.item() * batch.num_graphs
            optimizer.step()
        return loss_all / len(self.train_loader.dataset)

    def _test(self, model, loader):
        model.eval()
        error = 0
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                label_out, _ = model(batch, 0)
            error += self.batch_error_func(label_out, batch.y, self.std)
        return self.all_error_func(error / len(loader.dataset))

    def run(self):
        model = models.DirlNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                               min_lr=0.00001)
        best_val_error = float('inf')
        best_model_dic = None
        for e in range(self.epoch):
            train_loss = self._train(model, optimizer, e)
            val_error = self._test(model, self.val_loader)
            scheduler.step(val_error)
            if val_error <= best_val_error:
                best_val_error = val_error
                best_model_dic = copy.deepcopy(model.state_dict())
            print(e, train_loss, val_error)
        model = models.DirlNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        model.load_state_dict(best_model_dic)
        test_error = self._test(model, self.test_loader)
        self._save(test_error, best_model_dic)


class CfrIswTrainer(Trainer):
    def __init__(self, task, scenario, **kw):
        super(CfrIswTrainer, self).__init__(task, scenario, **kw)
        self.alpha = kw.get('alpha')
        assert self.alpha is not None

        self.name = 'CfrIsw' + str(self.alpha) + '==>' + self.name

    def _train(self, R, L, D, disc_func, optimizer):
        R.train()
        L.train()
        D.train()
        bias_iter = iter(self.train_loader)
        unbias_iter = iter(self.test_loader)
        loss_rl, loss_d = 0, 0
        for _ in range(len(bias_iter)):
            optimizer.zero_grad()
            bias_batch = bias_iter.next().to(self.device)
            unbias_batch = unbias_iter.next().to(self.device)

            bias_repr = R(bias_batch)
            unbias_repr = R(unbias_batch)

            disc_loss = disc_func(global_mean_pool(bias_repr, bias_batch.batch),
                                  global_mean_pool(unbias_repr, unbias_batch.batch))
            with torch.no_grad():
                weights = 1 / (2 * torch.exp(D(bias_batch, bias_repr.data)[:, 1]))
            label_loss = F.mse_loss(L(bias_batch, bias_repr), bias_batch.y, reduction='none')
            label_loss = (label_loss * weights).mean()
            loss = self.alpha * disc_loss + label_loss
            loss.backward()
            loss_rl += loss.item() * bias_batch.num_graphs
            optimizer.step()

            optimizer.zero_grad()
            loss = F.nll_loss(
                torch.cat((D(bias_batch, bias_repr.data), D(unbias_batch, unbias_repr.data))),
                torch.cat((torch.ones(bias_batch.num_graphs),
                           torch.zeros(unbias_batch.num_graphs))).to(torch.int64).to(self.device)
            )
            loss.backward()
            loss_d += loss.item() * (bias_batch.num_graphs + unbias_batch.num_graphs)
            optimizer.step()
        loss_rl = loss_rl / len(self.train_loader.dataset)
        loss_d = loss_d / (len(self.train_loader.dataset) + len(self.test_loader.dataset))
        return loss_rl, loss_d

    def _test(self, R, L, loader):
        R.eval()
        L.eval()
        error = 0
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = L(batch, R(batch))
            error += self.batch_error_func(pred, batch.y, self.std)
        return self.all_error_func(error / len(loader.dataset))

    def run(self):
        R = models.CausalFeatureNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        L = models.CausalRegressNet(self.h_dim, self.e_dim, self.times).to(self.device)
        D = models.CausalClassifyNet(self.h_dim, self.e_dim, self.times).to(self.device)
        optimizer = torch.optim.Adam(list(R.parameters()) + list(L.parameters()) + list(D.parameters()), lr=self.lr)
        # There will not be val_error for D training part, thus we use two optimizer.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                             min_lr=0.00001)
        best_val_error = float('inf')
        best_model_dic = None
        disc_func = SamplesLoss('sinkhorn')
        for e in range(self.epoch):
            train_loss_lr, train_loss_d = self._train(R, L, D, disc_func, optimizer)
            val_error = self._test(R, L, self.val_loader)
            scheduler.step(val_error)
            if val_error <= best_val_error:
                best_val_error = val_error
                best_model_dic = (copy.deepcopy(R.state_dict()),
                                  copy.deepcopy(L.state_dict()),
                                  copy.deepcopy(D.state_dict()))
            print(e, train_loss_lr, train_loss_d, val_error)
        R = models.CausalFeatureNet(self.i_dim, self.h_dim, self.e_dim, self.times).to(self.device)
        L = models.CausalRegressNet(self.h_dim, self.e_dim, self.times).to(self.device)
        D = models.CausalClassifyNet(self.h_dim, self.e_dim, self.times).to(self.device)
        R.load_state_dict(best_model_dic[0])
        L.load_state_dict(best_model_dic[1])
        D.load_state_dict(best_model_dic[2])
        test_error = self._test(R, L, self.test_loader)
        self._save(test_error, best_model_dic)

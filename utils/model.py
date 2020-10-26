""" This module contains necessary parts of image classifier model training, including model
loading and model training.
"""
import os.path as osp
import time
import warnings
from collections import defaultdict as dd
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import utils.common as common_utils
from datasets import get_dataset
from models import zoo


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def get_criterion(criterion_type: str, reduction: str = 'mean', weight: torch.Tensor = None):
    if criterion_type == 'MSE':
        if weight is not None:
            warnings.warn("MSELoss does not support weighted loss, weight ignored", Warning)
        return nn.MSELoss(reduction=reduction)
    elif criterion_type == 'CE':
        # Cross Entropy
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    elif criterion_type == 'SCE':
        return lambda x, y: soft_cross_entropy(x, y, weight)
    elif criterion_type == 'L1':
        if weight is not None:
            warnings.warn("MSELoss does not support weighted loss, weight ignored", Warning)
        return nn.L1Loss(reduction=reduction)
    elif criterion_type == 'NLL':
        return nn.NLLLoss(weight=weight, reduction=reduction)
    elif criterion_type == 'BCE':
        return nn.BCELoss(weight=weight, reduction=reduction)
    elif criterion_type == 'SmoothL1':
        if weight is not None:
            warnings.warn("MSELoss does not support weighted loss, weight ignored", Warning)
        return nn.SmoothL1Loss(reduction=reduction)
    else:
        raise NotImplementedError


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, test_loader, criterion, device, epoch=0., silent=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    t_end = time.time()

    acc = 100. * correct / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total))

    return test_loss, acc


# Preserved for backward compatibility
def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        common_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step()
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model


class Model(object):
    def __init__(
            self,
            model_dir: str,
            model_arch: str,
            input_size: int,
            dataset: str,
            epochs: int,
            device: torch.device = None,
            complexity: int = -1,
            lr: float = 0.01,
            momentum: float = 0.5,
            log_interval: int = 50,
            resume: str = None,
            lr_step: int = 60,
            lr_gamma: float = 0.1,
            pretrained: str = None,
            weighted_loss: bool = False,
            optimizer_choice: str = 'sgdm',
            batch_size: int = 64,
            num_workers: int = 10,
            train_criterion: str = 'CE',
            test_criterion: str = 'CE',
            reduction: str = 'mean',
            **kwargs
    ):
        self.testset = get_dataset(dataset, input_size, False)
        num_classes = len(self.testset.classes)
        sample = self.testset[0][0]
        if len(sample.shape) < 3:
            channel = 1
        else:
            channel = sample.shape[0]

        if complexity == -1 and channel == 3:
            self.model = zoo.get_net(model_arch, input_size, pretrained, num_classes=num_classes).to(device)
        else:
            # net that contains channel/complexity parameters must include arbitrary param discard.
            self.model = zoo.get_net(model_arch, input_size, pretrained,
                                     num_classes=num_classes, channel=channel, complexity=complexity).to(device)
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_choice, lr, momentum)
        if resume is not None:
            if osp.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=str(device))
                self.epoch = checkpoint['epoch']
                self.best_test_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            self.best_test_acc = 0.0
            self.epoch = 1
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)
        if not osp.exists(model_dir):
            os.mkdir(model_dir)
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_criterion = get_criterion(train_criterion, reduction)
        self.test_criterion = get_criterion(test_criterion, reduction)
        self.weighted_loss = weighted_loss
        if device is not None:
            self.device = device
        else:
            raise AssertionError('Device should be initialize from the beginning.')
        self.log_interval = log_interval
        self.total_epochs = epochs
        self.train_loader = None
        self.test_loader = None

    def __step(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_loss_batch = 0
        acc = 0.
        epoch_size = len(self.train_loader.dataset)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.train_criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if len(targets.size()) == 2:
                # Labels could be a posterior probability distribution. Use argmax as a proxy.
                target_probs, target_labels = targets.max(1)
            else:
                target_labels = targets
            correct += predicted.eq(target_labels).sum().item()

            prog = total / epoch_size
            exact_epoch = self.epoch + prog - 1
            acc = 100. * correct / total
            train_loss_batch = train_loss / total
            if (batch_idx + 1) % self.log_interval == 0:
                print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                    exact_epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                                 100. * batch_idx / len(self.train_loader),
                    loss.item(), acc, correct, total))
            acc = 100. * correct / total
        return train_loss_batch, acc

    def test(self, testset: Dataset = None, silent=False):
        self.model.eval()
        test_loss = 0.
        correct = 0
        total = 0
        t_start = time.time()
        if testset is None:
            test_loader = self.test_loader
        else:
            test_loader = DataLoader(testset, self.batch_size, shuffle=False, num_workers=self.num_workers)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.test_criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        test_loss /= total

        if not silent:
            print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(self.epoch, test_loss, acc,
                                                                                 correct, total))

        return test_loss, acc

    def train(self, trainset: Dataset, suffix: str = ''):
        best_train_acc, train_acc = -1., -1.
        best_test_acc, test_acc, test_loss = -1., -1., -1.
        if self.epoch == self.total_epochs or self.epoch == 1:
            start_epoch = 1
        else:
            # resume
            best_test_acc = self.best_test_acc
            start_epoch = self.epoch

        run_id = str(datetime.now())

        if self.weighted_loss:
            if not isinstance(trainset.samples[0][1], int):
                raise Exception('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]),
                                                                                          int))
            class_to_count = dd(int)
            for _, y in trainset.samples:
                class_to_count[y] += 1
            class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
            print('=> counts per class: ', class_sample_count)
            weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
            weight = weight.to(self.device)
            print('=> using weights: ', weight)
            self.train_criterion.register_buffer('weight', weight)

        # log initialization
        log_path = osp.join(self.model_dir, 'train{}.log.tsv'.format(suffix))
        if not osp.exists(log_path):
            with open(log_path, 'w') as wf:
                columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
                wf.write('\t'.join(columns) + '\n')
        model_path = osp.join(self.model_dir, 'checkpoint{}.pth.tar'.format(suffix))
        self.train_loader = DataLoader(trainset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
        if self.testset is not None:
            self.test_loader = DataLoader(self.testset, shuffle=False,
                                          batch_size=self.batch_size, num_workers=self.num_workers)
        for epoch in range(start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            train_loss, train_acc = self.__step()
            self.scheduler.step()
            best_train_acc = max(best_train_acc, train_acc)
            if self.test_loader is not None:
                test_loss, test_acc = self.test()
                best_test_acc = max(best_test_acc, test_acc)

                # Checkpoint
            if test_acc >= best_test_acc:
                state = {
                    'epoch': epoch,
                    'arch': self.model.__class__,
                    'state_dict': self.model.state_dict(),
                    'best_acc': test_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'created_on': str(datetime.now()),
                }
                torch.save(state, model_path)
                self.best_test_acc = test_acc
            with open(log_path, 'a') as af:
                train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
                af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    def __call__(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model.forward(x)

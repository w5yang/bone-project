import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adversary import Adversary
import torchattacks
from torchattacks.attack import Attack
from utils.common import Subset, Complement, QueryWrapper, parser_dealer, setup_seed


class TargetBalance(Adversary):
    def __init__(self, args, **kwargs):
        super(TargetBalance, self).__init__(args)

        self.verbose = False
        if kwargs['test']:
            perm = np.random.permutation(len(self.sampler.dataset))
            self.sampler.dataset = Subset(self.sampler.dataset, [i for i in perm[:10000]])
            self.verbose = True
        self.num_classes = len(self.model.testset.classes)
        self.penalty_standard = []
        self.penalty_matrix = np.zeros((self.num_classes, self.num_classes))
        self.temp_penalty = np.zeros((self.num_classes, self.num_classes))
        self.statistic = np.zeros((self.num_classes, self.num_classes), dtype=int)
        self.max_iter = 200
        self.overshoot = .02
        self.max_select_per_direction = 100
        self.synthetic = []
        self.attacker: Attack = torchattacks.CW(self.model.model, c=0.5, steps=50)
        self.attacker.set_attack_mode('targeted')

    def punish(self, i, j):
        self.statistic[i, j] += 1
        k = int(len(self.penalty_standard) * self.statistic[i, j] / self.max_select_per_direction) - 1
        if self.verbose:
            print('Punish i={}, j={}, k={}'.format(i, j, k))
            if k > len(self.penalty_standard):
                print('Overflow')
        if k < len(self.penalty_standard):
            self.temp_penalty[i, j] = self.penalty_standard[k] - self.penalty_standard[0] - self.penalty_matrix[i, j]
            assert self.temp_penalty[i, j] > 0
            # self.penalty_matrix[i, j] = self.penalty_standard[k] - self.penalty_standard[0]
        else:
            # self.penalty_matrix[i, j] = np.inf
            self.temp_penalty[i, j] = np.inf
        if self.verbose:
            if self.penalty_matrix[i, j] == np.inf:
                print('({}, {}) are never selected.'.format(i, j))
        # repunish tensor that has been already punished.
        repunish = []
        for index in self.punished:
            if self.direction_tensor[index].tolist() == [i, j]:
                repunish.append(index)
        self.punished.difference_update(repunish)

    def update_penalty_matrix(self):
        self.penalty_matrix += self.temp_penalty

    def choose(self, budget: int):
        selecting_pool = Complement(self.sampler)
        perturbations = []
        directions = []
        self.temp_penalty = np.zeros([self.num_classes, self.num_classes])
        with tqdm(total=len(selecting_pool)) as pbar:
            for x_t, y_t in DataLoader(selecting_pool, self.model.batch_size, True):
                y = self.model(x_t.to(self.device)).argmax(1)
                pert_batch_list = []
                for i in range(self.num_classes):
                    x_adv = self.attacker(x_t, i * torch.ones(x_t.size(0), dtype=int))
                    x_pert = x_adv.to(self.device) - x_t.to(self.device)
                    x_pert = torch.norm(x_pert.flatten(1), dim=1)
                    for j in range(x_t.size(0)):
                        x_pert[j] += self.penalty_matrix[y[j], i]
                    pert_batch_list.append(x_pert.clone().detach())
                pert_batch = torch.stack(pert_batch_list, dim=1)
                max_pert_batch, max_dest_batch = pert_batch.max(1)
                dir_batch = torch.stack([y, max_dest_batch], dim=1)
                perturbations.append(max_pert_batch)
                directions.append(dir_batch)
                pbar.update(x_t.size(0))

        perturbation_tensor = torch.cat(perturbations, dim=0)
        perturbation_array = perturbation_tensor.cpu().numpy()
        perturbation_clone = perturbation_array.copy()
        self.penalty_standard = perturbation_tensor.cpu().numpy()
        self.penalty_standard.sort()
        self.direction_tensor = torch.cat(directions, dim=0).cpu().numpy()
        sorted_indices = perturbation_array.argsort()
        chosen = set()
        current = 0
        self.punished = set()
        while len(chosen) < budget:
            current_index = sorted_indices[current]
            i, j = self.direction_tensor[current_index]
            if self.temp_penalty[i, j] <= 0 or current_index in self.punished:
                chosen.add(current_index)
                self.punish(i, j)
                current += 1
            else:
                perturbation_array[current] = perturbation_clone[current_index] + self.temp_penalty[i, j]
                sorted_indices = perturbation_array.argsort()
                self.punished.add(current_index)
        real_chosen = selecting_pool.convert_indices(chosen)
        labels = self.query_tensor(QueryWrapper(self.sampler.dataset, real_chosen))
        self.sampler.extend(real_chosen, labels)
        self.update_penalty_matrix()


def main():
    parser = argparse.ArgumentParser(description='Train a model in a distillation manner.')
    # Required arguments
    parser_dealer(parser, 'blackbox')
    parser_dealer(parser, 'sampling')
    parser_dealer(parser, 'train')
    parser_dealer(parser, 'common')
    parser.add_argument('--method', metavar='M', type=str, help='Determine sample/synthetic method.',
                        default='CW', choices=['CW', 'untargeted'])
    args = parser.parse_args()
    setup_seed(0)
    adversary = TargetBalance(args, test=True)
    selecting = np.random.permutation(len(adversary.sampler.dataset))[:1000]
    labels = adversary.query_tensor(QueryWrapper(adversary.sampler.dataset, selecting))
    adversary.sampler.extend(selecting, labels)
    adversary.train()
    for i in range(10):
        adversary.choose(500)
        # todo this line should be rechecked once the test argument is removed.
        np.save(os.path.join(adversary.model.model_dir, 'selected_.npy'),
                adversary.sampler.dataset.convert_indices(adversary.sampler.indices))
        # np.save(os.path.join(adversary.model.model_dir, 'synthetic_.npy'),
        #         [(tensor.numpy(), result.numpy()) for tensor, result in adversary.synthetic])

        np.save(os.path.join(adversary.model.model_dir, 'statistic_matrix_.npy'), adversary.statistic)
        adversary.train()



if __name__ == '__main__':
    main()

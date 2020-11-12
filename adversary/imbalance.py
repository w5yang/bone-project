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


class Imbalance(Adversary):
    def __init__(self, args, **kwargs):
        super(Imbalance, self).__init__(args)

        self.verbose = False
        if kwargs['test']:
            perm = np.random.permutation(len(self.sampler.dataset))
            self.sampler.dataset = Subset(self.sampler.dataset, [i for i in perm[:10000]])
            self.verbose = True
        self.num_classes = len(self.model.testset.classes)
        # self.penalty_standard = []
        # self.penalty_matrix = torch.zeros((self.sampler.num_classes, self.sampler.num_classes))
        # self.temp_penalty = torch.zeros((self.sampler.num_classes, self.sampler.num_classes))
        self.statistic = torch.zeros((self.num_classes, self.num_classes), dtype=int)
        self.max_iter = 200
        self.overshoot = .02
        self.max_select_per_direction = 100
        self.synthetic = []
        self.attacker: Attack = torchattacks.CW(self.model.model, c=1.0)
        self.attacker.set_attack_mode('least_likely')

    def choose(self, budget: int):
        selecting_pool = Complement(self.sampler)
        # perturbations = []
        directions = []
        # self.temp_penalty = torch.zeros((self.sampler.num_classes, self.sampler.num_classes))
        perturbation_list = []
        with tqdm(total=len(selecting_pool)) as pbar:
            for x_t, y_t in DataLoader(selecting_pool, self.model.batch_size, True):
                y = self.model(x_t.to(self.device)).argmax(1)
                x_adv = self.attacker(x_t, y)
                x_pert = x_adv.to(self.device) - x_t.to(self.device)
                x_pert = torch.norm(x_pert.flatten(1), dim=1)
                y_adv = self.model(x_adv.to(self.device)).argmax(1)
                # pert_batch_list.append(x_pert.clone().detach())
                # pert_batch = torch.stack(pert_batch_list, dim=1)
                # max_pert_batch, max_dest_batch = pert_batch.max(1)
                directions.append(torch.stack([y, y_adv], dim=1))
                perturbation_list.append(x_pert)
                # directions.append(dir_batch)
                pbar.update(x_t.size(0))

        perturbation_tensor = torch.cat(perturbation_list, dim=0)
        # perturbation_clone = perturbation_tensor.detach().clone()
        # self.penalty_standard = perturbation_tensor.detach().clone()
        direction_tensor = torch.cat(directions, dim=0)
        sorted_indices = perturbation_tensor.argsort()
        # chosen = set()
        # current = 0
        # self.punished = set()
        # while len(chosen) < budget:
        #     current_index = sorted_indices[current]
        #     i, j = self.direction_tensor[current_index]
        #     if self.temp_penalty[i, j] <= 0 or current_index in self.punished:
        #         chosen.add(current_index)
        #         self.punish(i, j)
        #         current += 1
        #     else:
        #         perturbation_tensor[current] = perturbation_clone[current_index] + self.temp_penalty[i, j]
        #         sorted_indices = perturbation_tensor.argsort()
        #         self.punished.add(current_index)
        for index in sorted_indices[:budget]:
            i, j = direction_tensor[index]
            self.statistic[i, j] += 1
        real_chosen = selecting_pool.convert_indices(sorted_indices[:budget])
        labels = self.query_tensor(QueryWrapper(self.sampler.dataset, real_chosen))
        self.sampler.extend(real_chosen, labels)


def main():
    parser = argparse.ArgumentParser(description='Train a model in a distillation manner.')
    # Required arguments
    parser_dealer(parser, 'blackbox')
    parser_dealer(parser, 'sampling')
    parser_dealer(parser, 'train')
    parser_dealer(parser, 'common')
    parser.add_argument('--method', metavar='M', type=str, help='Determine sample/synthetic method.',
                        default='CW', choices=['CW'])
    args = parser.parse_args()
    setup_seed(0)
    adversary = Imbalance(args, test=True)
    selecting = np.random.permutation(len(adversary.sampler.dataset))[:1000]
    labels = adversary.query_tensor(QueryWrapper(adversary.sampler.dataset, selecting))
    adversary.sampler.extend(selecting, labels)
    adversary.train()
    for i in range(10):
        adversary.choose(500)
        # todo this line should be rechecked once the test argument is removed.
        np.save(os.path.join(adversary.model.model_dir, 'selected_.npy'), adversary.sampler.dataset.convert_indices(adversary.sampler.indices))
        np.save(os.path.join(adversary.model.model_dir, 'synthetic_.npy'),
                [(tensor.numpy(), result.numpy()) for tensor, result in adversary.synthetic])
        adversary.train()
    np.save(os.path.join(adversary.model.model_dir, 'statistic_matrix_.npy'), adversary.statistic)


if __name__ == '__main__':
    main()

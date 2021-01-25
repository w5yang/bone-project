import argparse
import os
from typing import Union, Tuple

import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from adversary import Adversary
from utils.common import parser_dealer, QueryWrapper, Subset


class BalanceAdversary(Adversary):
    def __init__(self, args, **kwargs):
        super(BalanceAdversary, self).__init__(args)
        self.verbose = False
        if kwargs["test"]:
            perm = np.random.permutation(len(self.sampler.dataset))
            self.sampler.dataset = Subset(
                self.sampler.dataset, [i for i in perm[:20000]]
            )
            self.verbose = True
        self.num_classes = len(self.model.testset.classes)
        self.penalty_standard = []
        self.penalty_matrix = np.zeros(
            (self.sampler.num_classes, self.sampler.num_classes)
        )
        self.temp_penalty = np.zeros(
            (self.sampler.num_classes, self.sampler.num_classes)
        )
        self.statistic = np.zeros((self.sampler.num_classes, self.sampler.num_classes))
        self.max_iter = 200
        self.overshoot = 0.02
        self.max_select_per_direction = 100
        self.synthetic = []

    def get_perturbation(
        self, item: Union[int, torch.Tensor]
    ) -> Tuple[float, Tuple[int, int]]:
        """

        :param item: refers to current indices
        :return: penalty, (i -> j)
        """
        self.model.model.eval()
        softmax = torch.nn.Softmax(1)
        if isinstance(item, int):
            sample: torch.Tensor = self.sampler.dataset[item][0].to(self.device)
        elif isinstance(item, torch.Tensor):
            sample = item
        else:
            raise TypeError
        sample = sample.unsqueeze(0)

        penalty = torch.zeros(
            [self.num_classes], device=self.device, dtype=torch.float32
        )
        pert_sample = sample.clone()
        x = pert_sample.clone().requires_grad_()
        output = self.model.forward(x)
        with torch.no_grad():
            probs = softmax(output)
            top_prob, i = probs.max(1)
            i = i.item()
        loop_j = 0
        pert_label = i
        penalty[i] = np.inf
        output[0, i].backward(retain_graph=True)
        origin_grad = x.grad.clone().detach()
        for j in range(self.num_classes):
            # i -> j
            if j == i:
                continue
            r_tot = torch.zeros_like(sample, device=self.device)
            while pert_label == i and loop_j < self.max_iter:
                zero_gradients(x)
                output[0, j].backward(retain_graph=True)
                current_grad = x.grad.clone().detach()
                with torch.no_grad():
                    w_i = current_grad - origin_grad
                    prob_delta = probs[0, j] - top_prob
                    pert_i = torch.abs(prob_delta) / torch.norm(w_i.flatten())
                    r_i = (pert_i + 1e-4) * w_i / torch.norm(w_i)
                    r_tot = r_i + r_tot
                    pert_sample = pert_sample + (1 + self.overshoot) * r_tot
                x = pert_sample.clone().requires_grad_()
                output = self.model.forward(x)
                pert_label = output.max(1)[1]
                loop_j += 1
            if pert_label != j:  # target unreachable
                penalty[j] = np.inf
            else:
                penalty[j] = torch.norm(r_tot).item() + self.penalty_matrix[i, j]
            pert_label = i
        index = penalty.argsort()[0]
        return penalty[index].item(), (i, index)

    def deepfool_synthetic(
        self, item: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        self.model.model.eval()
        softmax = torch.nn.Softmax(1)
        if isinstance(item, int):
            sample: torch.Tensor = self.sampler.dataset[item][0].to(self.device)
        elif isinstance(item, torch.Tensor):
            sample = item
        else:
            raise TypeError
        sample = sample.unsqueeze(0)

        penalty = torch.zeros(
            [self.num_classes], device=self.device, dtype=torch.float32
        )
        pert_sample = sample.clone()
        x = pert_sample.clone().requires_grad_()
        output = self.model.forward(x)
        with torch.no_grad():
            probs = softmax(output)
            top_prob, i = probs.max(1)
            i = i.item()
        loop_j = 0
        pert_label = i
        penalty[i] = np.inf
        output[0, i].backward(retain_graph=True)
        origin_grad = x.grad.clone().detach()
        perturbations = []
        for j in range(self.num_classes):
            # i -> j
            if j == i:
                perturbations.append(torch.zeros_like(sample))
                continue
            r_tot = torch.zeros_like(sample, device=self.device)
            while pert_label == i and loop_j < self.max_iter:
                zero_gradients(x)
                output[0, j].backward(retain_graph=True)
                current_grad = x.grad.clone().detach()
                with torch.no_grad():
                    w_i = current_grad - origin_grad
                    prob_delta = probs[0, j] - top_prob
                    pert_i = torch.abs(prob_delta) / torch.norm(w_i.flatten())
                    r_i = (pert_i + 1e-4) * w_i / torch.norm(w_i)
                    r_tot = r_i + r_tot
                    pert_sample = pert_sample + (1 + self.overshoot) * r_tot
                x = pert_sample.clone().requires_grad_()
                output = self.model.forward(x)
                pert_label = output.max(1)[1]
                loop_j += 1
            if pert_label != j:  # target unreachable
                penalty[j] = np.inf
                perturbations.append(pert_sample.squeeze(0).cpu())
            else:
                penalty[j] = torch.norm(r_tot).item() + self.penalty_matrix[i, j]
                perturbations.append(pert_sample.squeeze(0).cpu())
            pert_label = i
        index = penalty.argsort()[0]
        return perturbations[index], penalty[index].item(), (i, index)

    def punish(self, i: int, j: int):
        self.statistic[i][j] += 1
        k = (
            int(
                len(self.penalty_standard)
                * self.statistic[i, j]
                / self.max_select_per_direction
            )
            - 1
        )
        if self.verbose:
            print("Punish i={}, j={}, k={}".format(i, j, k))
            if k > len(self.sampler.dataset):
                print("Overflow")
        if k < len(self.penalty_standard):
            self.temp_penalty[i, j] = (
                self.penalty_standard[k]
                - self.penalty_standard[0]
                - self.penalty_matrix[i, j]
            )
            self.penalty_matrix[i][j] = (
                self.penalty_standard[k] - self.penalty_standard[0]
            )
        else:
            self.penalty_matrix[i][j] = np.inf
        if self.verbose:
            if self.penalty_matrix[i, j] == np.inf:
                print("({}, {}) are never selected.".format(i, j))

    def choose(self, method: str, budget: int):
        if method == "random":
            super(BalanceAdversary, self).choose(method, budget)
        elif method == "balance":
            selecting_pool = set(list(range(len(self.sampler.dataset))))
            selecting_pool.difference_update(self.sampler.indices)
            selecting_pool = list(selecting_pool)
            self.temp_penalty = np.zeros(
                [self.sampler.num_classes, self.sampler.num_classes]
            )
            penalty_list = []
            direction_list = []
            for i in tqdm(selecting_pool):
                penalty, direction = self.get_perturbation(i)
                penalty_list.append(penalty)
                direction_list.append(direction)
            self.penalty_standard = penalty_list[:]
            self.penalty_standard.sort()
            arg_penalty = np.argsort(penalty_list)
            punished = set()  # For temp punishment
            chosen = set()
            current = 0
            while len(chosen) < budget:
                current_index = arg_penalty[current]
                i, j = direction_list[current_index]
                if self.temp_penalty[i, j] == 0 or current_index in punished:
                    chosen.add(current_index)
                    self.punish(i, j)
                    current += 1
                else:
                    penalty_list[current_index] += self.temp_penalty[i, j]
                    arg_penalty = np.argsort(penalty_list)
                    punished.add(current_index)
            real_chosen = [selecting_pool[i] for i in chosen]
            labels = self.query_tensor(QueryWrapper(self.sampler.dataset, real_chosen))
            self.sampler.extend(real_chosen, labels)
        elif method == "balance_pert":
            selecting_pool = set(list(range(len(self.sampler.dataset))))
            selecting_pool.difference_update(self.sampler.indices)
            selecting_pool = list(selecting_pool)
            self.temp_penalty = np.zeros(
                [self.sampler.num_classes, self.sampler.num_classes]
            )
            penalty_list = []
            direction_list = []
            pert_samples = []
            for i in tqdm(selecting_pool):
                pert_sample, penalty, direction = self.deepfool_synthetic(i)
                penalty_list.append(penalty)
                direction_list.append(direction)
                pert_samples.append(pert_sample)
            self.penalty_standard = penalty_list[:]
            self.penalty_standard.sort()
            arg_penalty = np.argsort(penalty_list)
            punished = set()  # For temp punishment
            chosen = set()
            current = 0
            while len(chosen) < budget:
                current_index = arg_penalty[current]
                i, j = direction_list[current_index]
                if self.temp_penalty[i, j] == 0 and current_index not in punished:
                    chosen.add(current_index)
                    self.punish(i, j)
                    current += 1
                elif current_index not in punished:
                    chosen.add(current_index)
                    self.punish(i, j)
                    current += 1
                else:
                    penalty_list[current_index] += self.temp_penalty[i, j]
                    arg_penalty = np.argsort(penalty_list)
                    punished.add(current_index)
            real_chosen = [pert_samples[i] for i in chosen]

            labels = self.query_tensor(real_chosen)
            self.sampler.extend([selecting_pool[i] for i in chosen])
            self.synthetic.extend(list(zip(real_chosen, labels)))
        elif method == "deepfool":
            selecting_pool = set(list(range(len(self.sampler.dataset))))
            selecting_pool.difference_update(self.sampler.indices)
            selecting_pool = list(selecting_pool)
            self.temp_penalty = np.zeros(
                [self.sampler.num_classes, self.sampler.num_classes]
            )
            penalty_list = []
            direction_list = []
            for i in tqdm(selecting_pool):
                penalty, direction = self.get_perturbation(i)
                penalty_list.append(penalty)
                direction_list.append(direction)
            self.penalty_standard = penalty_list[:]
            self.penalty_standard.sort()
            arg_penalty = np.argsort(penalty_list)
            chosen = set()
            for i in range(budget):
                current_index = arg_penalty[i]
                i, j = direction_list[current_index]
                self.statistic[i, j] += 1
                chosen.add(current_index)
            real_chosen = [selecting_pool[i] for i in chosen]
            labels = self.query_tensor(QueryWrapper(self.sampler.dataset, real_chosen))
            self.sampler.extend(real_chosen, labels)
        elif method == "deepfool_pert":
            selecting_pool = set(list(range(len(self.sampler.dataset))))
            selecting_pool.difference_update(self.sampler.indices)
            selecting_pool = list(selecting_pool)
            self.temp_penalty = np.zeros(
                [self.sampler.num_classes, self.sampler.num_classes]
            )
            penalty_list = []
            direction_list = []
            pert_samples = []
            for i in tqdm(selecting_pool):
                pert_sample, penalty, direction = self.deepfool_synthetic(i)
                penalty_list.append(penalty)
                pert_samples.append(pert_sample)
                # direction_list.append(direction)
            # self.penalty_standard = penalty_list[:]
            # self.penalty_standard.sort()
            arg_penalty = np.argsort(penalty_list)
            real_chosen = [pert_samples[i] for i in arg_penalty[:budget]]
            # punished = set()  # For temp punishment
            # chosen = set()
            # current = 0
            # while len(chosen) < budget:
            #     current_index = arg_penalty[current]
            #     i, j = direction_list[current_index]
            #     if self.temp_penalty[i, j] == 0 and current_index not in punished:
            #         chosen.add(current_index)
            #         self.punish(i, j)
            #         current += 1
            #     elif current_index not in punished:
            #         chosen.add(current_index)
            #         self.punish(i, j)
            #         current += 1
            #     else:
            #         penalty_list[current_index] += self.temp_penalty[i, j]
            #         arg_penalty = np.argsort(penalty_list)
            #         punished.add(current_index)
            # real_chosen = [pert_samples[i] for i in chosen]

            labels = self.query_tensor(real_chosen)
            self.sampler.extend([selecting_pool[i] for i in arg_penalty[:budget]])
            self.synthetic.extend(list(zip(real_chosen, labels)))
        else:
            raise NotImplementedError

    def train(self, samples=None):
        if samples is None:
            trainset = ConcatDataset([self.sampler, self.synthetic])
            self.model.train(trainset, suffix=".{}".format(len(trainset)))
        else:
            self.model.train(samples, suffix=".{}".format(len(samples)))


def main():
    parser = argparse.ArgumentParser(
        description="Train a model in a distillation manner."
    )
    # Required arguments
    parser_dealer(parser, "blackbox")
    parser_dealer(parser, "sampling")
    parser_dealer(parser, "train")
    parser_dealer(parser, "common")
    parser.add_argument(
        "--method",
        metavar="M",
        type=str,
        help="Determine sample/synthetic method.",
        default="balance",
        choices=["balance", "deepfool", "random", "balance_pert", "deepfool_pert"],
    )
    args = parser.parse_args()
    adversary = BalanceAdversary(args, test=True)
    adversary.choose("random", 5000)
    # adversary.train()
    for i in range(2):
        adversary.choose(args.method, 5000)
        np.save(
            os.path.join(adversary.model.model_dir, "selected_.npy"),
            adversary.sampler.indices,
        )
        np.save(
            os.path.join(adversary.model.model_dir, "synthetic_.npy"),
            [
                (tensor.numpy(), result.numpy())
                for tensor, result in adversary.synthetic
            ],
        )
        adversary.train()
    np.save(
        os.path.join(adversary.model.model_dir, "statistic_matrix_.npy"),
        adversary.statistic,
    )


if __name__ == "__main__":
    main()

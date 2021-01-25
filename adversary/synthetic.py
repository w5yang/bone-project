import argparse
import numpy as np

import torchattacks
from adversary import Adversary
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader

from utils.common import Subset, PureSubset, parser_dealer
from utils.model import Model


class SyntheticAdversary(Adversary):
    """An implementation of Papernot. Parameters may vary."""

    def __init__(self, args, **kwargs):
        super(SyntheticAdversary, self).__init__(args, method="random", **kwargs)
        self.exponential_increase_iteration = args.exp_iter
        self.linear_increase_iteration = args.lin_iter
        self.epsilon = args.epsilon
        # self.initial = args.init
        # self.budget_per_iteration = args.budget
        self.total_data: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def train(self, samples=None):
        self.model = Model(device=self.device, **self.params)
        if samples is None:
            self.model.train(self.total_data, suffix=".{}".format(len(self.total_data)))
        else:
            self.model.train(samples, suffix=".{}".format(len(samples)))

    def choose(self, budget: int) -> None:
        """This serves as different role from adversary. This is only the starting point where a batch of random chosen
        samples is needed.

        Args:
            budget: Initial random choose budget.

        Returns: None

        """
        super(SyntheticAdversary, self).choose(budget)
        for x, y in self.sampler:
            self.total_data.append((x, y))

    def synthetic(self, indices: List[int], mode: str, method="FGSM") -> None:
        """Synthetic adversary samples, currently only support FGSM.

        Args:
            indices: synthetic base image index.
            mode: attack mode, make attacker targeted or untargetd.
            method: FGSM

        Returns: None

        """
        assert method == "FGSM"
        attacker = torchattacks.FGSM(self.model.model, eps=self.epsilon)
        attacker.set_attack_mode(mode)
        synthetic = []
        base = PureSubset(self.total_data, indices)
        loader = DataLoader(
            base,
            self.model.batch_size,
            shuffle=False,
            num_workers=self.model.num_workers,
        )
        for x_t, y_t in loader:
            x_m = attacker(x_t, y_t)
            y_m = self.blackbox(x_m.to(self.device))
            for i in range(x_t.size(0)):
                synthetic.append((x_m[i], y_m[i]))


def main():
    parser = argparse.ArgumentParser(
        description="Train a model in a distillation manner."
    )
    # Required arguments
    parser_dealer(parser, "blackbox")
    parser_dealer(parser, "sampling")
    parser_dealer(parser, "train")
    parser_dealer(parser, "common")
    args = parser.parse_args()
    parser.add_argument(
        "--exp-iter",
        metavar="EXP",
        type=int,
        help="Iteration of exponential increase.",
        default=2,
    )
    parser.add_argument(
        "--lin-iter",
        metavar="EXP",
        type=int,
        help="Iteration of linear increase.",
        default=50,
    )
    parser.add_argument(
        "--epsilon",
        metavar="EPS",
        type=float,
        help="Parameters for FGSM attack.",
        default=0.1,
    )
    parser.add_argument(
        "--init-budget",
        metavar="INIT",
        type=int,
        help="Initial random budget.",
        default=500,
    )
    parser.add_argument(
        "--iter-budget",
        metavar="INCR",
        type=int,
        help="Budget for each linear increase selection.",
        default=500,
    )
    adversary = SyntheticAdversary(args)
    adversary.choose(args.init_budget)
    adversary.train()
    targeted = False
    for i in range(args.exp_iter):
        # The synthetic samples increase exponentially.
        indices = list(range(len(adversary.total_data)))
        if targeted:
            adversary.synthetic(indices, mode="targeted")
        else:
            adversary.synthetic(indices, mode="original")
        adversary.train()
        targeted = not targeted

    for i in range(args.lin_iter):
        indices = np.random.choice(
            range(len(adversary.total_data)), args.iter_budget, replace=False
        )
        if targeted:
            adversary.synthetic(indices, mode="targeted")
        else:
            adversary.synthetic(indices, mode="original")
        adversary.train()
        targeted = not targeted


if __name__ == "__main__":
    main()

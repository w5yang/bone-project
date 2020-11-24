import argparse
from typing import List, Tuple, Union, Sequence

import numpy as np
from overloading import overload
from torch import Tensor
from torch.utils.data import Dataset

from utils.common import (
    query,
    parser_dealer,
    device_dealer,
    PseudoBlackbox,
    Subset,
    QueryWrapper,
    QuerySubset,
)
from utils.model import Model
from victim.blackbox import Blackbox


class Adversary(object):
    def __init__(self, args, **kwargs):
        # keep a backup of args
        self.args = args
        self.params = vars(args)
        self.device = device_dealer(args.device_id)
        self.model = Model(device=self.device, **self.params)
        self.sampler = Subset.from_args(args)
        self.method = args.method
        if args.pseudoblackbox and args.bydataset:
            # This may work, but is poorly designed.
            self.blackbox = PseudoBlackbox(self.sampler.dataset, args.argmax)
        elif args.pseudoblackbox:
            self.blackbox = PseudoBlackbox(args.blackbox_dir, args.argmax)
        elif args.argmax:
            self.blackbox = Blackbox.from_modeldir(
                args.blackbox_dir, device=self.device, topk=1, rounding=0
            )
        elif args.topk != 0:
            self.blackbox = Blackbox.from_modeldir(
                args.blackbox_dir, device=self.device, topk=args.topk
            )
        else:
            self.blackbox = Blackbox.from_modeldir(
                args.blackbox_dir, device=self.device
            )

    def query_tensor(self, x: Union[Sequence[Tensor], QueryWrapper]) -> List:
        if isinstance(self.blackbox, PseudoBlackbox):
            raise NotImplementedError
        else:
            return query(self.blackbox, x, device=self.device)

    def train(self, samples=None):
        self.model = Model(device=self.device, **self.params)
        if samples is None:
            self.model.train(self.sampler, suffix=".{}".format(len(self.sampler)))
        else:
            self.model.train(samples, suffix=".{}".format(len(samples)))

    def choose(self, budget: int):
        if self.method == "random":
            unselected = list(
                set(range(len(self.sampler.dataset))).difference(self.sampler.indices)
            )
            selecting = np.random.permutation(unselected)[:budget]
            if isinstance(self.sampler, QuerySubset):
                results = self.query_tensor(
                    QueryWrapper(self.sampler.dataset, selecting)
                )
                self.sampler.extend(selecting, results)
        else:
            raise NotImplementedError


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
        "--method",
        metavar="M",
        type=str,
        help="Determine sample/synthetic method.",
        default="balance",
        choices=["random"],
    )
    adversary = Adversary(args)
    adversary.choose(2000)
    adversary.train()


if __name__ == "__main__":
    main()

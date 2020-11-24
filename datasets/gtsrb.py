#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp

from torchvision.datasets.folder import ImageFolder

import configs.config as cfg


class GTSRB(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, "GTSRB", "Final_Training")
        if not osp.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it from {}.".format(
                    root,
                    "http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset",
                )
            )

        # Initialize ImageFolder
        super().__init__(
            root=osp.join(root, "Images"),
            transform=transform,
            target_transform=target_transform,
        )

        self.root = root
        trainning_size = len(self.samples)
        self.read_test(osp.join(cfg.DATASET_ROOT, "GTSRB", "Final_Test", "Images"))

        self.partition_to_idxs = self.get_partition_to_idxs(trainning_size)
        self.pruned_idxs = self.partition_to_idxs["train" if train else "test"]

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print(
            "=> done loading {} ({}) with {} examples".format(
                self.__class__.__name__, "train" if train else "test", len(self.samples)
            )
        )

    def read_test(self, folder):
        with open(osp.join(folder, "GT-final_test.csv")) as f:
            f.readline()
            for line in f:
                image, _, _, _, _, _, _, label = line.strip().split(";")
                path = osp.join(folder, image)
                self.samples.append((path, int(label)))
                self.targets.append(int(label))

    def get_partition_to_idxs(self, training_size):
        partition_to_idxs = {"train": [], "test": []}

        # ----------------- Create mapping: filename -> 'train' / 'test'
        # There are two files: a) images.txt containing: <imageid> <filepath>
        #            b) train_test_split.txt containing: <imageid> <0/1>

        # imageid_to_filename = dict()
        # with open(osp.join(self.root, 'images.txt')) as f:
        #     for line in f:
        #         imageid, filepath = line.strip().split()
        #         _, filename = osp.split(filepath)
        #         imageid_to_filename[imageid] = filename
        # filename_to_imageid = {v: k for k, v in imageid_to_filename.items()}

        # imageid_to_partition = dict()
        # with open(osp.join(self.root, 'train_test_split.txt')) as f:
        #     for line in f:
        #         imageid, split = line.strip().split()
        #         imageid_to_partition[imageid] = 'train' if int(split) else 'test'

        # # Loop through each sample and group based on partition
        # for idx, (filepath, _) in enumerate(self.samples):
        #     _, filename = osp.split(filepath)
        #     imageid = filename_to_imageid[filename]
        #     partition_to_idxs[imageid_to_partition[imageid]].append(idx)

        for i in range(training_size):
            partition_to_idxs["train"].append(i)
        for i in range(training_size, len(self.samples)):
            partition_to_idxs["test"].append(i)
        return partition_to_idxs

import os.path as osp

import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import configs.config as cfg
from typing import Callable
from PIL import Image


class ImageNet64(Dataset):
    """Define Imagenet downscale dataset, the experiments of this paper only covered the downscaled version of ImageNet
    Dataset, and we only select the first one of the total 10 training batch as ImageNet queryset.

    """

    def __init__(
        self,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        root = osp.join(cfg.DATASET_ROOT, "Imagenet64")
        self.transform = transform
        self.target_transform = target_transform
        if not osp.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it from {}.".format(
                    root, "http://image-net.org/download-images"
                )
            )
        if train:
            with open(osp.join(root, "train_data_batch_1"), "rb") as f:
                dump = pickle.load(f)

        else:
            with open(osp.join(root, "val_data"), "rb") as f:
                dump = pickle.load(f)
            # with open(osp.join(root, 'eval.pickle'), 'rb') as f:
            #     querylabel = pickle.load(f)
            #     dump['labels'] = [label.argmax() for label in querylabel]

        data = dump["data"]
        labels = dump["labels"]
        img_size = 64
        img_size2 = img_size * img_size
        self.classes = ["{}".format(i) for i in range(1000)]
        data = np.dstack(
            (
                data[:, :img_size2],
                data[:, img_size2 : 2 * img_size2],
                data[:, 2 * img_size2 :],
            )
        )
        data = data.reshape((data.shape[0], img_size, img_size, 3))

        self.samples = [(data[i], labels[i]) for i in range(data.shape[0])]

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.samples[index]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

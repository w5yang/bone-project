from torch.utils import data
from torch.utils.data import Dataset, dataset

from datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from datasets.gtsrb import GTSRB
from datasets.imagenet64 import ImageNet64

from models import sized_transforms, imagenet_transform


def get_dataset(dataset_name: str, input_size: int, train: bool) -> Dataset:
    if 'imagenet' in dataset_name.lower():
        transform = imagenet_transform[input_size]
    else:
        transform = sized_transforms[input_size]
    return eval(dataset_name)(train, transform)

from torch.utils.data import Dataset

from datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from datasets.gtsrb import GTSRB
from datasets.imagenet64 import ImageNet64

from models import sized_transforms


def get_dataset(dataset_name: str, input_size: int, train: bool) -> Dataset:
    transform = sized_transforms[input_size]
    return eval(dataset_name)(train, transform)

import os
import os.path as osp

from typing import List, Tuple, Union, Sequence
# from overloading import overload
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import datasets

from victim.blackbox import Blackbox
from models import zoo, sized_transforms
from torch import Tensor
from torch import device as Device

import os
import warnings

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser, Namespace


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        num_classes (int): Should be specified if the num_classes is not the attribute of dataset.

    """

    def __init__(self, dataset: Dataset, indices: Sequence, num_classes: int = -1, state_path: str = None):
        self.state_path = state_path
        self.dataset = dataset
        if hasattr(dataset, 'dataset_name'):
            self.dataset_name = dataset.dataset_name
        else:
            self.dataset_name = dataset.__class__.__name__
        if len(set(indices)) == len(indices):
            self.indices = list(indices)
        else:
            warnings.warn('There is duplication in initial indices!', UserWarning)
            # maintain the order of indices
            tempset = set()
            self.indices = []
            for i in indices:
                if i not in tempset:
                    self.indices.append(i)
                    tempset.add(i)
        if num_classes == -1:
            if hasattr(dataset, 'classes'):
                self.num_classes = len(dataset.classes)
            else:
                raise Exception("The num_classes should be specified if the dataset doesn't own classes attribute.")
        else:
            self.num_classes = num_classes
        if hasattr(dataset, "samples"):
            self.__samples = dataset.samples
        if hasattr(dataset, "classes"):
            self.classes = dataset.classes

    def __getitem__(self, idx: int):
        # leave all the data representation to the original dataset.
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def samples(self, idx: int):
        if hasattr(self, 'samples'):
            return self.__samples[self.indices[idx]]
        else:
            return None

    def extend(self, indices, *args, **kwargs):
        warned = False
        tempset = set(self.indices)
        for i in indices:
            if i not in tempset:
                self.indices.append(i)
                tempset.add(i)
            elif not warned:
                warnings.warn('There is duplication in extended list')
                warned = True

    @classmethod
    def from_args(cls, args: Namespace):
        """ This method get params from parser and return neccesary state.

            :param args: Parser that has parsed state params.
            """
        try:
            sampleset = args.sampleset
            load_state = args.load_state
            state_suffix = args.state_suffix
            budget = args.partial
            directory = args.model_dir
            input_size = args.input_size
        except AttributeError:
            raise AttributeError("State params or model directory/input size not specified.")
        dataset = initial_dataset(sampleset, input_size, train=True)
        if load_state:
            return load_selection_state(dataset, directory, state_suffix, budget)
        else:
            # default returns querysubset
            return QuerySubset(dataset, [], [], len(dataset.classes))

    def save_state(self, state_path: str = None, suffix: str = None, budget: int = -1):
        if state_path is None and self.state_path is None:
            raise Exception("State path not specified.")
        if state_path is None:
            save_selection_state(self, self.state_path, suffix, budget)
        else:
            save_selection_state(self, state_path, suffix, budget)


class Complement(Dataset):
    def __init__(self, subset: Subset):
        self.dataset = subset.dataset
        self.dataset_name = subset.dataset_name
        self.subset = subset

    def __len__(self):
        return len(self.dataset) - len(self.subset)

    def __getitem__(self, item):
        current = set([i for i in range(len(self.dataset))])
        current.difference_update(self.subset.indices)
        index = list(current)[item]
        return self.dataset[index]


class Queryset(Dataset):

    def __init__(self, dataset, labels):
        if hasattr(dataset, 'dataset_name'):
            self.dataset_name = dataset.dataset_name
        else:
            self.dataset_name = dataset.__class__.__name__
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.labels[idx]

    def __len__(self):
        return len(self.dataset)


class QuerySubset(Subset):
    def __init__(
            self,
            dataset: Dataset,
            indices: Sequence,
            labels: Sequence,
            num_classes: int = -1,
            state_path: str = None
    ):
        self.labels = list(labels)
        super(QuerySubset, self).__init__(dataset, indices, num_classes, state_path)
        self.__indices = self.indices[:]

    def __getitem__(self, item):
        return self.dataset[self.__indices[item]][0], self.labels[item]

    def __len__(self):
        return len(self.__indices)

    def extend(self, indices, new_labels=None, *args, **kwargs):
        if new_labels is None:
            tempset = set(self.indices)
            for i in indices:
                if i not in tempset:
                    self.indices.append(i)
                    tempset.add(i)
            return
        else:
            assert len(indices) == len(new_labels)
        warned = False
        tempset = set(self.indices)
        for i, label in zip(indices, new_labels):
            if i not in tempset:
                self.indices.append(i)
                self.__indices.append(i)
                self.labels.append(label)
                tempset.add(i)
            elif not warned:
                    warnings.warn('There is duplication in extended list')
                    warned = True


class PseudoBlackbox(object):
    def __init__(self, target: Union[str, Dataset], argmax: bool = False):
        """

        :param target:
        :param argmax:
        """
        if isinstance(target, str):
            with open(os.path.join(target, 'train.pickle'), 'rb') as f:
                self.train_results = pickle.load(f)
            with open(os.path.join(target, 'eval.pickle'), 'rb') as f:
                eval_results = pickle.load(f)
            self.eval_results = [r.argmax() for r in eval_results]
            self.is_dataset = False
        elif isinstance(target, Dataset):
            self.train_results = target
            self.is_dataset = True
        self.argmax = argmax

    def __call__(self, index: int, train: bool = True):
        if self.is_dataset:
            x = self.train_results[index][1]
        else:
            x = self.train_results[index]
        if train:
            if self.argmax:
                temp = x
                m = temp.argmax()
                value = torch.zeros_like(temp)
                value[m] = 1.0
                return value
            return x
        else:
            return self.eval_results[index]


class QueryWrapper(object):
    def __init__(self, dataset: Dataset, indices=None):
        self.dataset = dataset
        if hasattr(dataset, 'dataset_name'):
            self.dataset_name = dataset.dataset_name
        else:
            self.dataset_name = dataset.__class__.__name__
        if indices is None:
            self.indices = [i for i in range(len(dataset))]
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return self.dataset[i][0]


# typing
DatasetType = Union[Dataset, Subset, Queryset, Sequence[Tuple[Tensor, Tensor]]]
BlackboxType = Union[Blackbox, nn.Module]


def device_dealer(device_id: int) -> torch.device:
    if device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def parser_dealer(parser: ArgumentParser, option: str):
    """Add parameters to parser by common scene

    :param parser: param parser that needs specific scene of parameters.
    :param option: { 'sampling': this option collects sample set name, and load previous selected sample set state,
        'blackbox': this option collects blackbox info,
        'train': this option is used to train a generic model,
        'common': Read parameters that are commonly used.
    }
    """

    if option == 'sampling':
        parser.add_argument('sampleset', metavar='DS_NAME', type=str,
                            help='Name of sample dataset in active learning selecting algorithms')
        parser.add_argument('--load-state', action='store_true', default=False, help='Turn on if load state.')
        parser.add_argument('--state-suffix', metavar='SE', type=str,
                            help='load selected samples from sample set', required=False, default='')
        parser.add_argument('--partial', metavar='N', type=int, help='load partial set of sample set', default=-1)
    if option == 'blackbox':
        parser.add_argument('blackbox_dir', metavar='VIC_DIR', type=str,
                            help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
        parser.add_argument('--argmax', action='store_true', help='Only consider argmax labels', default=False)
        parser.add_argument('--pseudoblackbox', action='store_true', help='Load prequeried labels as blackbox',
                            default=False)
        parser.add_argument('--bydataset', action='store_true', help='Load prequeried labels as blackbox',
                            default=False)
        parser.add_argument('--topk', metavar='TK', type=int, help='iteration times',
                            default=0)
    if option == 'train':
        parser.add_argument('model_dir', metavar='MODEL_DIR', type=str,
                            help='Destination directory of model to be trained')
        parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
        parser.add_argument('input_size', metavar='MODEL_SIZE', type=int, help='The size of input image.',
                            choices=(32, 224))
        parser.add_argument('dataset', metavar='DS_NAME', type=str,
                            help='Name of test dataset. In the case of victim model training, '
                                 'this parameter refer to both training set and test set')
        # Optional arguments
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 100)')
        # This is only useful when the model support this complexity settings
        parser.add_argument('-x', '--complexity', type=int, default=-1, metavar='N',
                            help="Model conv channel size.")
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                            help='Step sizes for LR')
        parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                            help='LR Decay Rate')
        parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
        parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
        parser.add_argument('--optimizer-choice', type=str, help='Optimizer', default='sgdm',
                            choices=('sgd', 'sgdm', 'adam', 'adagrad'))
        parser.add_argument('--train-criterion', type=str, help='Loss Function of training process', default='SCE',
                            choices=['MSE', 'CE', 'L1', 'NLL', 'BCE', 'SmoothL1', 'SCE'])
        parser.add_argument('--test-criterion', type=str, help='Loss Function of test process', default='CE',
                            choices=['MSE', 'CE', 'L1', 'NLL', 'BCE', 'SmoothL1'])
        parser.add_argument('--reduction', type=str, help='Loss Function reduction type', default='mean',
                            choices=['mean', 'sum'])
    if option == 'common':
        parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('-d', '--device-id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
        parser.add_argument('-w', '--num-workers', metavar='N', type=int, help='# Worker threads to load data',
                            default=10)


def initial_dataset(dataset_name, input_size, train=True):
    dataset_initializer = getattr(datasets, dataset_name)
    transform = sized_transforms[input_size]
    dataset = dataset_initializer(train=train, transform=transform)
    return dataset


def query(
        blackbox: BlackboxType,
        samples: Sequence[Tensor],
        budget: int = -1,
        argmax: bool = False,
        batch_size: int = 1024,
        device: Device = Device('cpu'),
        topk: int = 0,
) -> List:
    """Query blackbox with a list of examples.

    :param blackbox: Blackbox type or nn Module
    :param samples: Query samples
    :param budget: If only part of samples are required to be queried, this parameter must be specified.
    :param argmax: Whether the query result should be argmaxed, default False.
    :param batch_size: Batch size of each query.
    :param device: Computing device
    :param topk: Truncation
    :return: List of queried results, specifically query labels.
    """
    results = []
    if budget == -1:
        budget = len(samples)
    with tqdm(total=budget) as pbar:
        for x_t in DataLoader(samples, batch_size, shuffle=False):
            with torch.no_grad():
                y_t = blackbox(x_t.to(device))
            if argmax:
                y_t = y_t.argmax(1)
            elif topk != 0:
                v, i = y_t.topk(topk, 1)
                y_t = torch.zeros_like(y_t).scatter(1, i, v)
            # unpack
            for i in range(x_t.size(0)):
                results.append(y_t[i].cpu())
            pbar.update(x_t.size(0))
    return results


def query_dataset(
        blackbox: Union[Blackbox, nn.Module, PseudoBlackbox],
        dataset: DatasetType,
        list_indices: List[int] = None,
        topk: int = 0,
        argmax: bool = False,
        batch_size: int = 1024,
        device: Device = Device('cpu')
) -> Queryset:
    """This dataset is designed to query blackbox for the whole dataset, and return the result as dataset type.

    :param list_indices: This parameter specifies the range of dataset to be queried.
    :param blackbox: Blackbox nn module
    :param dataset: Dataset typed data
    :param topk: return the vector with top-k labels
    :param argmax: return the class labels
    :param batch_size: batch size
    :param device: calculation device, CPU as default
    :return: queryset type results, can be used as dataset
    """
    if list_indices is None:
        if isinstance(blackbox, PseudoBlackbox):
            labels = [blackbox(i) for i in range(len(dataset))]
            return Queryset(dataset, labels)
        subset = dataset
    else:
        subset = Subset(dataset, list_indices)
        if isinstance(blackbox, PseudoBlackbox):
            labels = [blackbox(i) for i in list_indices]
            return Queryset(subset, labels)
    labels = []
    dataset_size = len(subset)
    loader = DataLoader(subset, batch_size, shuffle=False, num_workers=8)
    with tqdm(total=dataset_size) as pbar:
        for x_t, _ in loader:
            x_t = x_t.to(device)
            # for t, B in enumerate(range(0, dataset_size, batch_size)):
            # x_t = torch.stack(
            #     [dataset[i][0] for i in range(B, min(B + batch_size, dataset_size))]).to(
            #     device)
            with torch.no_grad():
                y_t = blackbox(x_t)
            if argmax:
                y_t = y_t.argmax(1)
            elif topk != 0:
                v, i = y_t.topk(topk, 1)
                y_t = torch.zeros_like(y_t).scatter(1, i, v)
            for i in range(x_t.size(0)):
                labels.append(y_t[i].cpu())
            pbar.update(x_t.size(0))
    return Queryset(subset, labels)


def load_transferset(path: str, topk: int = 0, argmax: bool = False) -> (List, int):
    assert os.path.exists(path)
    with open(path, 'rb') as rf:
        samples = pickle.load(rf)
    if argmax:
        results = [(item[0], int(item[1].argmax())) for item in samples]
    elif topk != 0:
        results = []
        for x, y in samples:
            values, indices = y.topk(topk)
            z = torch.zeros_like(y).scatter(0, indices, values)
            results.append((x, z))
    else:
        results = samples
    num_classes = samples[0][1].size(0)
    return results, num_classes


def save_selection_state(data: Subset, state_dir: str, suffix: str = "", budget: int = -1) -> None:
    """ All selection state is maintained by Subset class. A QuerySubset Class is acceptable as well.

    :param data: A subset object contains indices.
    :param state_dir: Specify the save dir.
    :param suffix: A suffix can be attached to the filename.
    :param budget: Specify the required save quantity.
    :return: None
    """
    if os.path.exists(state_dir):
        assert os.path.isdir(state_dir)
    else:
        os.mkdir(state_dir)
    if budget > 0:
        label_path = os.path.join(state_dir, 'labels{}.pickle'.format(suffix))
        selected_indices_list_path = os.path.join(state_dir, 'selected_indices{}.pickle'.format(suffix))
    else:
        label_path = os.path.join(state_dir, 'labels{}.pickle'.format(suffix))
        selected_indices_list_path = os.path.join(state_dir, 'selected_indices{}.pickle'.format(suffix))

    if hasattr(data, 'labels'):  # QuerySubset
        if os.path.exists(label_path):
            print('Override previous transferset => {}'.format(label_path))
        with open(label_path, 'wb') as tfp:
            pickle.dump(data.labels, tfp)
        print("=> selected {} samples written to {}".format(len(data), label_path))

    if os.path.exists(selected_indices_list_path):
        print("{} exists, override file.".format(selected_indices_list_path))
    with open(selected_indices_list_path, 'wb') as lfp:
        pickle.dump(data.indices, lfp)
    print("=> selected {} samples written to {}".format(len(data), selected_indices_list_path))


def load_selection_state(queryset, state_dir: str, selection_suffix: str = '', budget: int = -1):
    label_path = os.path.join(state_dir, 'labels{}.pickle'.format(selection_suffix))
    indices_list_path = os.path.join(state_dir, 'selected_indices{}.pickle'.format(selection_suffix))
    with open(indices_list_path, 'rb') as lf:
        indices = pickle.load(lf)
        assert isinstance(indices, List)
        if budget > 0:
            indices = indices[:budget]
        print("=> load selected {} sample indices from {}".format(len(indices), indices_list_path))
    if os.path.exists(label_path):
        with open(label_path, 'rb') as tf:
            labels = pickle.load(tf)
            assert isinstance(labels, List)
            if budget > 0:
                labels = labels[:budget]
            print("=> load selected {} samples from {}".format(len(labels), label_path))
        return QuerySubset(queryset, indices, labels, len(queryset.classes), state_path=state_dir)
    return Subset(queryset, indices, state_path=state_dir)


def save_npimg(array: np.ndarray, path: str) -> None:
    """ Save numpy array to image file.

    :param array: img array
    :param path: path including corresponding extension
    :return: None
    """
    img = Image.fromarray(array.squeeze())
    img.save(path)


def tensor_to_np(tensor: Tensor) -> np.ndarray:
    img = tensor.mul(255).byte()
    img = img.cpu()
    if len(img.shape) == 4:
        img.squeeze_(0)
    elif len(img.shape) == 3:
        pass
    else:
        raise ValueError
    img = img.numpy().transpose((1, 2, 0))
    return img


def load_img_dir(img_dir: str, transform=None) -> List[torch.tensor]:
    imgs = []
    if transform is None:
        transform = transforms.ToTensor()
    for file in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, file))
        imgs.append(img)
    return [transform(img) for img in imgs]


# This function unpack the image tensor out of dataset-like List
unpack = lambda x: [item[0] for item in x]


def naive_onehot(index: int, total: int) -> Tensor:
    x = torch.zeros([total], dtype=torch.float32)
    x[index] = 1.0
    return x


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

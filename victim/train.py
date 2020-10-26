import argparse
import os.path as osp
import os
from datetime import datetime
import json

import numpy as np

import torch
from torch.utils.data import Subset

import datasets
import utils.model as model_utils
from utils.common import parser_dealer, device_dealer
import models.zoo as zoo
import models


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    # The training dataset is automatically equals to testdataset's training part.
    parser_dealer(parser, 'train')
    parser.add_argument('--train-subset', type=int, help='Use a subset of train set', default=None)
    parser_dealer(parser, 'common')
    args = parser.parse_args()
    params = vars(args)
    device = device_dealer(args.device_id)
    # ----------- Set up dataset
    dataset_name = args.dataset
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    input_size = args.input_size

    # modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    # train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    # test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    transform = models.sized_transforms[input_size]
    trainset = dataset(train=True, transform=transform)
    testset = dataset(train=False, transform=transform)
    num_classes = len(trainset.classes)
    sample = testset[0][0]
    if len(sample.shape) <= 2:
        # 2 dimensional images
        channel = 1
    else:
        channel = sample.shape[0]
    params['channel'] = channel
    params['num_classes'] = num_classes

    if params['train_subset'] is not None:
        idxs = np.arange(len(trainset))
        ntrainsubset = params['train_subset']
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        trainset = Subset(trainset, idxs)

    # ----------- Set up model
    model_name = args.model_arch
    pretrained = args.pretrained
    complexity = args.complexity   # if a model does not contain differernt complexity variance, this param will
                                        # have NO effects.
    model = zoo.get_net(model_name, input_size, pretrained, num_classes=num_classes,
                        channel=channel, complexity=complexity)
    model = model.to(device)
    optimizer = model_utils.get_optimizer(model.parameters(), params['optimizer_choice'], **params)

    # ----------- Train
    out_path = args.model_dir
    model_utils.train_model(model, trainset, testset=testset, device=device, optimizer=optimizer, out_path=out_path,
                            **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()

"""
"""
import os.path as osp
import json
import numpy as np

import torch
import torch.nn.functional as F

from utils.type_checks import TypeCheck
import models.zoo as zoo

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class Blackbox(object):
    def __init__(self, model, device=None, topk=None, rounding=None):
        self.device = torch.device('cuda') if device is None else device
        self.topk = topk
        self.rounding = rounding

        self.__model = model.to(device)
        self.__model.eval()

        self.__call_count = 0

    @classmethod
    def from_modeldir(cls, model_dir, device=None, topk=None, rounding=None):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        input_size = params['input_size']
        num_classes = params['num_classes']
        channel = params['channel']
        complexity = params['complexity']

        # Instantiate the model
        model = zoo.get_net(model_arch, input_size, pretrained=None, num_classes=num_classes,
                            channel=channel, complexity=complexity)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model, device, topk, rounding)
        return blackbox

    def truncate_output(self, y_t_probs):
        if self.topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, self.topk)
            newy = torch.zeros_like(y_t_probs)
            if self.rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if self.rounding is not None:
            y_t_probs = torch.Tensor(np.round(y_t_probs.numpy(), decimals=self.rounding))

        return y_t_probs

    def __call__(self, query_input):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)

        with torch.no_grad():
            query_input = query_input.to(self.device)
            query_output = self.__model(query_input)
            self.__call_count += query_input.shape[0]

            query_output_probs = F.softmax(query_output, dim=1).cpu()

        query_output_probs = self.truncate_output(query_output_probs)
        return query_output_probs

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.loss = nn.CrossEntropyLoss()

    def forward(self, points):
        self.points_shape = points.shape[1:]
        self.logits = self.model(points.transpose(1, 2).contiguous())[0]
        output = self.logits
        return output

    def backward(self, logits, targets):
        """
        classfication loss backpropagation
        """
        loss = self.loss(logits, targets)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                grad_out[0].requires_grad_()
                self.grad_pool[key] = grad_out[0]

            return backward_hook

        for name, module in self.model.module.named_modules():
            if name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        normVal = torch.norm(grads.contiguous().view(grads.shape[0], -1), 2, 1)
        grad_weight = grads/normVal.contiguous().view(grads.shape[0], 1)
        return fmaps, grad_weight

    def clear(self):
        for handle in self.handlers:
            handle.remove()
        del self.model, self.points_shape, self.logits, self.fmap_pool, self.grad_pool, self.loss

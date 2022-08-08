from __future__ import print_function, division

import os
import sys
import cv2
import csv
import copy
import time
import numpy as np
import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import config as cf
from model import *




######################################
# gradCAM

class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            res = fmap * weight.data.expand_as(fmap)
            gcam += fmap * weight.data.expand_as(fmap)
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        if(gcam.max() != 0):
            gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        if(gcam.max() != 0):
            gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


class BackPropagation(PropagationBase):
    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

    def generate(self, target_layer):
        grads = self._find(self.all_grads, target_layer)
        gradients_as_arr = grads.data[0].numpy()[0]
        return gradients_as_arr

    def save(self, filename, data):
        abs_max = np.maximum(-1 * data.min(), data.max())
        data = data / abs_max * 127.0 + 127.0
        cv2.imwrite(filename, np.uint8(data))


class GuidedBackPropagation(BackPropagation):

    def _set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

def save_class_activation_on_image(org_img, activation_map, file_name):
    """
    @ func:
        Saves CAM(Class Activation Map) on the original image.

    @ args:
        org_img (PIL img): Original image
        activation_map : Numpy array of the activation map in grayscale (0~255)
        file_name : String for the file name of the exported image.
    """

    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)

    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)

    # Heatmap on picture
    org_img = cv2.resize(org_img, (224,224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255*img_with_heatmap))


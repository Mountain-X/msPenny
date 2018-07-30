from __future__ import division

import argparse
import csv
from collections import OrderedDict
import os
import sys

import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import numpy as np

sys.path.append(os.path.split(os.getcwd())[0])
from common import np_transforms
from common import utils
from common.center_bias_layer import CenterBiasLayer
from common.dataset import get_dataset_config, TestDataset
from common.loaders import TestLoader
from common.loss import KLD
from densesal import models

OUTDIR = "/home/tatsuyasuzuki/mspenny_saliency/pytorch-saliency-map/DenseSal/output/"


class saliencyPredictor(object):
    def __init__(self, path2snapshot, outdir = OUTDIR,
                 no_center_bias=True):
        self.path2snapshot = path2snapshot
        self.out_dir = outdir
        self.no_center_bias = no_center_bias
        self._loadModel()

        # for image preprocessing
        self.mean = None
        self.imgTransformer = None

        self.maps = np.load("maps.npz")['a']

    def _loadModel(self, arch='densesalbi3', saveWeights=False,
                   no_constant_term_of_center_bias=False):
        model = models[arch](pretrained=False).cuda()
        model = torch.nn.DataParallel(model).cuda()
        # model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model)

        if os.path.isfile(self.path2snapshot):
            print("=> loading checkpoint '{}'".format(self.path2snapshot))
            checkpoint = torch.load(self.path2snapshot)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            self.model = model
            if 'center_bias_state_dict' not in checkpoint:
                self.no_center_bias = True
            if not self.no_center_bias:
                center_bias_layer =\
                    CenterBiasLayer(bias=not no_constant_term_of_center_bias).cuda()
                center_bias_layer.load_state_dict(checkpoint['center_bias_state_dict'])
                center_bias_layer.eval()
                self.center_bias_layer = center_bias_layer
                if saveWeights:
                    weight = center_bias_layer.weight.data[0, 0, :, :].cpu().numpy()
                    bias = center_bias_layer.bias.data[0, 0, :, :].cpu().numpy()
                    out_weight_file = os.path.join(self.out_dir, 'weight_of_center_bias.png')
                    out_bias_file = os.path.join(self.out_dir, 'bias_term_of_center_bias.png')
                    scipy.misc.imsave(out_weight_file, weight)
                    scipy.misc.imsave(out_bias_file, bias)
        else:
            raise OSError("No checkpoint found at '{}'".format(self.resume))

    def _preprocessData(self, listOfimages, training_dataset_name='osie'):
        """
            @args : list of images should be in RGB order
        """
        if self.mean is None:
            self.mean = get_dataset_config(training_dataset_name).mean_bgr
        if self.imgTransformer is None:
            self.imgTransformer = np_transforms.Compose([
                np_transforms.Resize(scale_factor=1.0),
                np_transforms.ToTensor(),
            ])
        imgs = []
        for img in listOfimages:
            img = img[:, :, ::-1].astype(np.float32)
            img -= self.mean
            img = self.imgTransformer(img)
            imgs.append(img)
        return imgs

    def _getHightSaliencyDirection(salMaps):
        """
            Get the direction with the highest salience among the 12 divisions.

            @args : list of salMaps should have 6 ndarrays

            return : Horizontal angle (-180 ~ 180)

            角度のmapは__init__でmap.npzをロードする。仰角を変えた場合はこのファイルを変更する必要あり。

            *todo ペニー自身の顕著性を省く
        """

        n = len(salMaps)
        h = salMaps[0].shape[0]
        w = salMaps[0].shape[1]
        center = int(w/2)

        salMapsDiv = np.empty((n*2, h, center))
        for i in range(n):
            #split salieny map left and right
            salMapsDiv[i*2] = salMaps[i][:,0:center]
            salMapsDiv[i*2 + 1] = salMaps[i][:,center:]


        maxDivision = salMapsDiv.sum(axis=1).sum(axis=1).argmax()

        mapsIndex = maxDivision//2
        mapsLeftOrRight = maxDivision%2 # left = 0, right = 1

        mapsWidth = self.maps.shape[2]
        mapsHight = self.maps.shape[1]
        mapsLeftCenter = int(mapsWidth/4)
        mapsRightCenter = int(mapsLeftCenter*3)

        if mapsLeftOrRight==0:
            return self.maps[mapsIndex, int(mapsHight/2), mapsLeftCenter][0]
        elif mapsLeftOrRight==1:
            print(mapsIndex, mapsRightCenter)
            return self.maps[mapsIndex, int(mapsHight/2), mapsRightCenter][0]

    def predict(self, images, saveMaps=True):
        """
            @args : list of images should be in RGB order

            *todo 画像の保存はscipyだとmaxを255、minを0にスケーリングされるので変更。そもそも画像として保存しなくて良いかもしれない。
        """
        images = self._preprocessData(images)
        salMaps = []
        for i, inputs in enumerate(images):
            inputs.unsqueeze_(0)
            input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
            if self.no_center_bias:
                saliency_map = self.model(input_var)
                # return tensor whose size is (1, 1, H, W).
            else:
                saliency_map = self.center_bias_layer(self.model(input_var))
            saliency_map = saliency_map[0, 0, :, :].data.cpu().numpy()
            salMaps.append(saliency_map)

            if saveMaps:
                out_file = os.path.join(self.out_dir, str(i).zfill(4)+'.png')
                scipy.misc.imsave(out_file, saliency_map)
                print('- out_file: {0}'.format(out_file))


        return _getHightSaliencyDirection(salMaps)

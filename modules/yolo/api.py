# -*- coding: utf-8 -*-
#from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import sys
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
#from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools


'''
yolov3
'''

# yolo param
reso = 64 # it should be a multiple of 32 and greater than 32
confidence = 0.1
nms_thesh = 0.4
batch_size = 6

weightsfile = '/Users/reo911gt3/Desktop/mspenny/modules/yolo/yolov3.weights'
cfgfile = '/Users/reo911gt3/Desktop/mspenny/modules/yolo/cfg/yolov3.cfg'

# yolov3 load
num_classes = 80
classes = load_classes('/Users/reo911gt3/Desktop/mspenny/modules/yolo/data/coco.names')

#Set up the neural network
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

model.net_info["height"] = reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
CUDA = torch.cuda.is_available()
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))


    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(orig_im, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = orig_im[:,:,::-1].astype(np.uint8).copy()

    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def detect(frames):
    from PIL import Image

    # 画像が一枚の時に未対応
    imlist = [frames[i] for i in range(frames.shape[0])]

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]

    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]

    i = 0

    write = False
    # model(get_test_input(inp_dim, CUDA), CUDA)

    objs = {}

    for batch in im_batches:
        #load the image
        if CUDA:
            batch = batch.cuda()

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

    #        prediction = prediction[:,scale_indices]


        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence
        #clubbing these ops in one loop instead of two.
        #loops are slower than vectorised operations.

        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)


        if type(prediction) == int:
            i += 1
            continue

        prediction[:,0] += i*batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        return None, None, None

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2


    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    output_n = output.numpy()

    # print("---------------")
    # print("output")
    # print(output_n)
    # print("---------------")

    output_person = output_n[output_n[:,7]==0]
    if len(output_person) != 0:
        max_score_index = output_person[:, 5].argmax()
        img_num = int(output_person[max_score_index][0])
        c1 = tuple(output_person[max_score_index][1:3])
        c2 = tuple(output_person[max_score_index][3:5])
        max_score = output_person[max_score_index][5]
        center_x = int(c2[0] - c1[0])
        print(img_num, center_x, max_score)
    else:
        print("No human was detections")
        return None, None, None

    torch.cuda.empty_cache()

    return img_num, center_x, max_score

'''
/yolov3
'''


"""
api = falcon.API()
api.add_route('/12345', API_yolo3())

from wsgiref import simple_server
httpd = simple_server.make_server('192.168.11.29', 8000, api)
httpd.serve_forever()
"""

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
from DenseSal.densesal import models

import cv2
from PIL import Image
import time
import math

class saliencyPredictor(object):
    def __init__(self, path2snapshot,
                 no_center_bias=True):
        self.path2snapshot = path2snapshot
        self.no_center_bias = no_center_bias
        self._loadModel()

        # for image preprocessing
        self.mean = None
        self.imgTransformer = None

        self.maps = np.load("maps30.npz")['a']

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

    def getHightSaliencyDirection(self,salMaps):
        """
            Get the direction with the highest salience among the 12 divisions.

            @args : list of salMaps should have 6 ndarrays

            return : Horizontal angle (-180 ~ 180)

            角度のmapは__init__でmap.npzをロードする。仰角を変えた場合はこのファイルを変更する必要あり。（とりあえず30度で固定）

            *todo ペニー自身の顕著性が邪魔なら省く
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

        return salMaps




def get_f2e_npz(h, input_h):
    """
    魚眼レンズ画像から正距円筒図法への変換マップを保存する。セットアップ時にnpzファイルがなければ実行する。
    正距円筒図法の画素毎に、魚眼レンズ画像のどの画素をとってくれば良いかを持っておく。

    また、その逆の変換マップも保存する。
    魚眼レンズ画像の画素毎に、正距円筒図法のどの画素をとってくれば良いかを持っておく。
    h : 出力する正距円筒図法の高さ
    input_h : 入力する魚眼レンズ画像の高さ
    """
    print("compute f2e and e2f map")
    input_w = input_h*2
    w = h*2

    f2e_mapA = np.zeros((h,w,2)).astype(np.int)
    f2e_mapB = np.zeros((h,w,2)).astype(np.int)
    y, x = np.meshgrid(np.arange(h),np.arange(w))
    v_t = -180 + 360*x/(w)
    v_p = 90 - 180*y/(h-1)
    f2e_mapA[y,x,0],  f2e_mapA[y,x,1] = mapping_f2e(v_p, v_t, input_h)
    v_t = 360*x/(w)
    f2e_mapB[y,x,0],  f2e_mapB[y,x,1] = mapping_f2e(v_p, v_t, input_h)

    f_x, f_y = np.meshgrid(np.arange(input_w),np.arange(input_h))
    e_x, e_y = np.meshgrid(np.arange(w),np.arange(h))
    e2f_mapA = np.ones((input_h, input_w, 2)).astype(np.int)*(-1)
    e2f_mapB = np.ones((input_h, input_w, 2)).astype(np.int)*(-1)


    e2f_mapA[f2e_mapA[e_y, e_x, 0], f2e_mapA[e_y, e_x, 1], 0] = e_y
    e2f_mapA[f2e_mapA[e_y, e_x, 0], f2e_mapA[e_y, e_x, 1], 1] = e_x
    e2f_mapB[f2e_mapB[e_y, e_x, 0], f2e_mapB[e_y, e_x, 1], 0] = e_y
    e2f_mapB[f2e_mapB[e_y, e_x, 0], f2e_mapB[e_y, e_x, 1], 1] = e_x

    np.savez('f2e_'+str(input_h)+'_'+str(h)+'.npz', A = f2e_mapA, B = f2e_mapB)
    np.savez('e2f_'+str(input_h)+'_'+str(h)+'.npz', A = e2f_mapA, B = e2f_mapB)


def mapping_f2e(v_p, v_t, input_h):
    """
    ある正距円筒図法の画素は魚眼レンズ画像のどの画素に対応するかを計算する。
    v_p : phi（垂直方向）の配列。2次元ndarray
    v_t : theta（水平方向）の配列。2次元ndarray
    input_h : 魚眼レンズ画像の高さ

    参考
    http://paulbourke.net/dome/dualfish2sphere/
    """
    aperture = np.radians(105)

    R = int(input_h/2)
    theta = np.radians(v_t)
    phi = np.radians(v_p)

    Px = np.cos(phi)*np.cos(theta)
    Py = np.cos(phi)*np.sin(theta)
    Pz = np.sin(phi)

    r = R*np.arccos(Px/(np.sqrt(Px**2+Py**2+Pz**2)))/aperture
    Px = Px - 1
    th = np.arctan2(Pz, Px)

    Px[theta>0] = -Px[theta>0]

    x = (Py/np.sqrt(Py**2+Pz**2))*r
    y = -(Pz/np.sqrt(Py**2+Pz**2))*r

    x += R
    y += R
    mask = np.zeros_like(v_p)

    mask[x<0] = 1
    mask[x>=R*2] = 1
    mask[y<0] = 1
    mask[y>=R*2] = 1
    mask[np.isnan(x)] = 1
    mask[np.isnan(y)] = 1

    y = y.astype(np.int)
    x = x.astype(np.int)

    y[mask==1] = 0
    x[mask==1] = 0
    return y, x

# def mapping_e2f(f_x, f_y,f_h):
#逆の計算は未完成
#埋め込んだあとの画素が飛び飛びにならないためにはこれを作成し、roundするのが良い。
#現在はget_f2e_npzによるマップを逆向きに使用している。
#     return y, x



def f2e(np_image, setupper):
    """
    魚眼レンズ画像を受け取り、正距円筒図法画像を返す。
    np_image : 魚眼レンズ画像のndarray(RGB)
    setupper : Setup_extract_omni_imageのインスタンス
    """

#     setupper.f_h = np_image.shape[0]
#     setupper.f_w = np_image.shape[1]
    frame_A = np_image[0:setupper.f_h, 0:setupper.f_h]
    frame_B = np_image[0:int(setupper.f_w/2), int(setupper.f_w/2):setupper.f_w]


    odiA = np.zeros((setupper.e_h, setupper.e_w, 3))
    odiA[0:setupper.e_h, 0:setupper.e_w] = frame_A[setupper.f2e_mapA[0:setupper.e_h, 0:setupper.e_w, 0], setupper.f2e_mapA[0:setupper.e_h, 0:setupper.e_w, 1]]
    odiB = np.zeros((setupper.e_h, setupper.e_w, 3))
    odiB[0:setupper.e_h, 0:setupper.e_w] = frame_B[setupper.f2e_mapB[0:setupper.e_h, 0:setupper.e_w, 0], setupper.f2e_mapB[0:setupper.e_h, 0:setupper.e_w, 1]]
    return odiA, odiB

def e2f(equirectangularImages, setupper):
    """
    正距円筒図法画像を受け取り、魚眼レンズ画像を返す。
    equirectangularImages : 正距円筒図法のndarray(RGB)2つをまとめたリスト
    setupper : Setup_extract_omni_imageのインスタンス
    """



    dualFishEyeImageA = np.zeros((setupper.f_h, setupper.f_w))
    dualFishEyeImageA = equirectangularImages [0][setupper.e2f_mapA[:,:,0] , setupper.e2f_mapA [:,:,1]]
    dualFishEyeImageA[:, int(setupper.f_w/2)-1:-1] = equirectangularImages [1][setupper.e2f_mapB[:,:,0], setupper.e2f_mapB[:,:,1] ][:, 0:int(setupper.f_w/2)]

    return dualFishEyeImageA


class Setup_extract_omni_image():

    def __init__(self, extract_num = 6, extract_outputsize = 400,
                 f2e_size = 1000, f_h = 736, view_angle = 90, elevation_angle = 45):
        self.extract_num = extract_num
        self.extract_outputsize = extract_outputsize
        self.e_h = f2e_size
        self.e_w = self.e_h*2
        self.f_h = f_h
        self.f_w = self.f_h*2
        self.view_angle = view_angle
        self.elevation_angle = elevation_angle

        theta_a = math.radians(self.view_angle)
        phi_a = math.radians(self.view_angle)

        if self.extract_num == 6:
            camerasA = np.array([[np.radians(-55),np.radians(self.elevation_angle)],
                                 [np.radians(0),np.radians(self.elevation_angle)],
                                 [np.radians(55),np.radians(self.elevation_angle)]])
            camerasB = np.array([[np.radians(125),np.radians(self.elevation_angle)],
                                 [np.radians(180),np.radians(self.elevation_angle)],
                                 [np.radians(-125),np.radians(self.elevation_angle)]])
        else:
            self.extract_num = 6
            camerasA = np.array([[np.radians(-55),np.radians(self.elevation_angle)],
                                 [np.radians(0),np.radians(self.elevation_angle)],
                                 [np.radians(55),np.radians(self.elevation_angle)]])
            camerasB = np.array([[np.radians(125),np.radians(self.elevation_angle)],
                                 [np.radians(180),np.radians(self.elevation_angle)],
                                 [np.radians(-125),np.radians(self.elevation_angle)]])

        self.cpA = [ CameraPrm(
                     camera_angle=camerasA[t],
                     image_plane_size=(self.extract_outputsize, self.extract_outputsize),
                     view_angle=(theta_a, phi_a)
                     ) for t in range(camerasA.shape[0]) ]

        self.cpB = [ CameraPrm(
                    camera_angle=camerasB[t],
                    image_plane_size=(self.extract_outputsize, self.extract_outputsize),
                    view_angle=(theta_a, phi_a)
                    ) for t in range(camerasB.shape[0]) ]

        self.embedA = [Embedding(self.cpA[num], (self.e_w, self.e_h)) for num in range(len(self.cpA))]
        self.embedB = [Embedding(self.cpB[num], (self.e_w, self.e_h)) for num in range(len(self.cpB))]

        if not(os.path.isfile('f2e_'+str(self.f_h)+'_'+str(self.e_h)+'.npz')):
            get_f2e_npz(self.e_h, self.f_h)
        f2e_map = np.load('f2e_'+str(self.f_h)+'_'+str(self.e_h)+'.npz')
        self.f2e_mapA = f2e_map['A']
        self.f2e_mapB = f2e_map['B']

        if not(os.path.isfile('e2f_'+str(self.f_h)+'_'+str(self.e_h)+'.npz')):
            get_f2e_npz(self.e_h, self.f_h)
        e2f_map = np.load('e2f_'+str(self.f_h)+'_'+str(self.e_h)+'.npz')
        self.e2f_mapA = e2f_map['A']
        self.e2f_mapB = e2f_map['B']

        #顕著性マップをカラー（ヒートマップ）化するときに、背景を消せるようにマスクを用意する。generate_inv_extract_mask.pyを使用して作成する。
        self.mask = np.load('mask'+str(self.elevation_angle)+'_'+str(self.f_h)+'.npz')['mask']

    def get_angles(self):
        angles = np.empty((self.extract_num, self.extract_outputsize, self.extract_outputsize, 2))
        h = self.e_h
        w = self.e_w
        y, x = np.meshgrid(np.arange(h),np.arange(w))
        v_t = -180 + 360*x/(w)
        v_p = 90 - 180*y/(h-1)

        v_t = v_t.T
        v_p = v_p.T

        odi_angles = np.empty((h, w, 2))

        odi_angles[:, :, 0] = v_t
        odi_angles[:, :, 1] = v_p

        imcA = OmniImage(odi_angles)
        impA = [ imcA.extract(self.cpA[num]) for num in range(len(self.cpA)) ]
        impA = np.array(impA)

        imcB = OmniImage(odi_angles)
        impB = [ imcB.extract(self.cpB[num]) for num in range(len(self.cpB)) ]
        impB = np.array(impB)

        angles[0] = impA[0]
        angles[1] = impA[1]
        angles[2] = impA[2]
        angles[3] = impB[0]
        angles[4] = impB[1]
        angles[5] = impB[2]
        return angles


def imresize(im, sz):
    if np.amax(im) <= 1.0:
        im = im * 255
        scl = 255.0
    else:
        scl = 1
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize((sz[1], sz[0])))/scl


def rnd(x):
    if type(x) is np.ndarray:
        return (x+0.5).astype(np.int)
    else:
        return round(x)


def polar(cord):
    if cord.ndim == 1:
        P = np.linalg.norm(cord)
    else:
        P = np.linalg.norm(cord, axis=0)
    phi = np.arcsin(cord[2] / P)
    theta_positive = np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta_negative = - np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta = (cord[1] > 0) * theta_negative + (cord[1] <= 0) * theta_positive
    return [theta, phi]


# r = [lower, upper]
def limit_values(x, r, xcopy=1):
    if xcopy == 1:
        ret = x.copy()
    else:
        ret = x
    ret[ret<r[0]] = r[0]
    ret[ret>r[1]] = r[1]
    return ret


# calculating polar cordinates of image plane in omni-directional image using camera parameters
class CameraPrm:
    # camera_angle, view_angle: [horizontal, vertical]
    # L: distance from camera to image plane
    def __init__(self, camera_angle, image_plane_size=None, view_angle=None, L=None):
        # camera direction (in radians) [horizontal, vertical]
        self.camera_angle = camera_angle
        # view_angle: angle of view in radians [horizontal, vertical]
        # image_plane_size: [image width, image height]
        if view_angle is None:
            self.image_plane_size = image_plane_size
            self.L = L
            self.view_angle = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L))
        elif image_plane_size is None:
            self.view_angle = view_angle
            self.L = L
            self.image_plane_size = 2.0 * L * np.tan(np.array(view_angle) / 2.0)
        else:
            self.image_plane_size = image_plane_size
            self.view_angle = view_angle
            L = (np.array(image_plane_size) / 2.0) / np.tan(np.array(view_angle) / 2.0)
            if rnd(L[0]) != rnd(L[1]):
                print('Warning: image_plane_size and view_angle are not matched.')
                va = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L[0]))
                ips = 2.0 * L[0] * np.tan(np.array(view_angle) / 2.0)
                print('image_plane_size should be (' + str(ips[0]) + ', ' + str(ips[1]) +
                      '), or view_angle should be (' +  str(math.degrees(va[0])) + ', ' + str(math.degrees(va[1])) + ').' )
                return
            else:
                self.L = L[0]

        # unit vector of cameara direction
        self.nc = np.array([
                            np.cos(camera_angle[1]) * np.cos(camera_angle[0]),
                            -np.cos(camera_angle[1]) * np.sin(camera_angle[0]),
                            np.sin(camera_angle[1])
                            ])
        # center of image plane
        self.c0 = self.L * self.nc

        # unit vector (xn, yn) in image plane
        self.xn = np.array([
                            -np.sin(camera_angle[0]),
                            -np.cos(camera_angle[0]),
                            0
                            ])
        self.yn = np.array([
                            -np.sin(camera_angle[1]) * np.cos(camera_angle[0]),
                            np.sin(camera_angle[1]) * np.sin(camera_angle[0]),
                            np.cos(camera_angle[1])
                            ])
        # meshgrid in image plane
        [c1, r1] = np.meshgrid(np.arange(0, rnd(self.image_plane_size[0])),
                               np.arange(0, rnd(self.image_plane_size[1])))

        # 2d-cordinates in image [xp, yp]
        img_cord = [c1 - self.image_plane_size[0] / 2.0,
                    -r1 + self.image_plane_size[1] / 2.0]

        # 3d-cordinatess in image plane [px, py, pz]
        self.p = self.get_3Dcordinate(img_cord)
        # polar cordinates in image plane [theta, phi]
        self.polar_omni_cord = polar(self.p)

    def get_3Dcordinate(self, img_cord):
        xp = img_cord[0]
        yp = img_cord[1]
        if type(xp) is np.ndarray: # xp, yp: array
            return xp * self.xn.reshape((3,1,1)) + yp * self.yn.reshape((3,1,1)) + np.ones(xp.shape) * self.c0.reshape((3,1,1))
        else: # xp, yp: scalars
            return xp * self.xn + yp * self.yn + self.c0


# omni-drectional image
class OmniImage:
    def __init__(self, omni_image):
        if type(omni_image) is str:
            self.omni_image = plt.imread(omni_image)
        else:
            self.omni_image = omni_image

    def extract(self, camera_prm):

        # 2d-cordinates in omni-directional image [c2, r2]
        c2 = (camera_prm.polar_omni_cord[0] / (2.0 * np.pi) + 1.0 / 2.0) * self.omni_image.shape[1] - 0.5
        r2 = (-camera_prm.polar_omni_cord[1] / np.pi + 1.0/2.0) * self.omni_image.shape[0] - 0.5
        # [c2_int, r2_int] = [rnd(c2), rnd(r2)]
        c2_int = limit_values(rnd(c2), (0, self.omni_image.shape[1]-1), 0)
        r2_int = limit_values(rnd(r2), (0, self.omni_image.shape[0]-1), 0)
        # self.omni_cord = [c2, r2]
        return self.omni_image[r2_int, c2_int]

# embedding extracted image in omni-directional image
class Embedding:
    # omni_size: [omni_image width, omni_image height]
    def __init__(self, camera_prm, omni_size):
        [c_omni, r_omni] = np.meshgrid(np.arange(omni_size[0]), np.arange(omni_size[1]))
        theta = (2.0 * c_omni / float(omni_size[0]-1) - 1.0) * np.pi
        phi = (0.5 - r_omni / float(omni_size[1]-1)) * np.pi
        pn = np.array([
                np.cos(phi) * np.cos(theta),
                -np.cos(phi) * np.sin(theta),
                np.sin(phi)
            ])
        pn = pn.transpose(1,2,0)

        # camera parameters
        L = camera_prm.L
        nc = camera_prm.nc
        xn = camera_prm.xn
        yn = camera_prm.yn
        theta_c = camera_prm.camera_angle[0]
        phi_c = camera_prm.camera_angle[1]
        theta_a = camera_prm.view_angle[0]
        phi_a = camera_prm.view_angle[1]
        w1 = camera_prm.image_plane_size[0]
        h1 = camera_prm.image_plane_size[1]

        # True: inside image (candidates), False: outside image
        cos_alpha = np.dot(pn, nc)
        mask = cos_alpha >= 2 * L / np.sqrt(w1**2 + h1**2 + 4*L**2) # circle

        r = np.zeros((omni_size[1], omni_size[0]))
        xp = np.zeros((omni_size[1], omni_size[0]))
        yp = np.zeros((omni_size[1], omni_size[0]))
        r[mask == True] = L / np.dot(pn[mask == True], nc)
        xp[mask == True] = r[mask == True] * np.dot(pn[mask == True], xn)
        yp[mask == True] = r[mask == True] * np.dot(pn[mask == True], yn)

        # True: inside image, False: outside image
        mask = (mask == True) & (xp > -w1/2.0) & (xp < w1/2.0) & (yp > -h1/2.0) & (yp < h1/2.0)
        r[mask == False] = 0
        xp[mask == False] = 0
        yp[mask == False] = 0

        self.camera_prm = camera_prm
        self.mask = mask
        self.r = r
        self.xp = xp
        self.yp = yp

        input_h  = 400
        input_w = 400
        [r1, c1] = np.array([input_h/2.0 - self.yp - 0.5, input_w/2.0 + self.xp - 0.5]) * self.mask
        self.r1_int = limit_values(rnd(r1), (0, input_h-1), 0)
        self.c1_int = limit_values(rnd(c1), (0, input_w-1), 0)

    # embed extracted image in omni image
    def embed(self, imp):
        # 2D cordinates in extracted image

        if imp.ndim == 3:
            ret = imp[self.r1_int, self.c1_int] * self.mask.reshape(self.mask.shape[0], self.mask.shape[1], -1)
        else:
            ret = imp[self.r1_int, self.c1_int] * self.mask

        return ret


def extract_omni_image(dualFishEyeImage, setupper):
    """
    魚眼レンズ画像から複数枚の平面画像を抽出して返す。

    dualFishEyeImage : 魚眼レンズ画像のndarray(RGB)
    setupper : Setup_extract_omni_imageクラスのインスタンス（変換のパラメータを渡す）
    """
    omni_im = np.empty((setupper.extract_num, setupper.extract_outputsize,
                        setupper.extract_outputsize, 3)).astype(np.uint8)

    odiA, odiB = f2e(dualFishEyeImage, setupper)
    imcA = OmniImage(odiA)
    for num in range(len(setupper.cpA)):
        omni_im[num] = imcA.extract(setupper.cpA[num]).astype(np.uint8)
    imcB = OmniImage(odiB)
    for num in range(len(setupper.cpB)):
        omni_im[len(setupper.cpA) + num] = imcB.extract(setupper.cpB[num]).astype(np.uint8)

    return omni_im

def inverse_extract_omni_image(planarImages, originalDualFishEyeImage, setupper, overlaid_weight1 = 0.3, overlaid_weight2 = 0.7):
    """
    複数枚の平面画像を魚眼レンズ画像に戻す。元の画像に対して重ねて表示させることも可能。

    planarImages : 平面画像のndarray。shapeは（枚数, 縦, 横, RGB）
    originalDualFishEyeImage : 元の魚眼レンズ画像のndarray(RGB)。
    setupper : Setup_extract_omni_imageクラスのインスタンス（変換のパラメータを渡す）
    overlaid_weight1 : 画像を重ねて表示させるときの魚眼レンズ画像の重み。
    overlaid_weight2 : 画像を重ねて表示させるときの顕著性画像の重み。

    return
    dualFishEyeImage : 元の画像に顕著性画像を重ねた画像のndarray(縦, 横, RGB)
    time1 : 終了時の時刻
    dualFishEyeSalMaps : 顕著性画像のndarray(縦, 横, RGB)


    """
    equirectanglarImageAs = [setupper.embedA[num].embed(planarImages[num]) for num in range(len(setupper.cpA))] #顕著性画像を平面から正距円筒図法に変換（A側3枚）
    equirectanglarImageA = np.array(equirectanglarImageAs).max(axis=0)
    equirectanglarImageBs = [setupper.embedB[num].embed(planarImages[num+3]) for num in range(len(setupper.cpB))]   #顕著性画像を平面から正距円筒図法に変換（B側3枚）
    equirectanglarImageB = np.array(equirectanglarImageBs).max(axis=0)
    dualFishEyeSalMaps = e2f([equirectanglarImageA, equirectanglarImageB], setupper)    #顕著性画像を正距円筒図法から魚眼レンズ画像に変換
    dualFishEyeSalMaps = change_salmap_color(dualFishEyeSalMaps, setupper)  #顕著性画像を白黒からカラーに変換
    dualFishEyeImage = cv2.addWeighted(originalDualFishEyeImage, overlaid_weight1, dualFishEyeSalMaps, overlaid_weight2, 0) #元画像と顕著性画像を重ねる
    time1 = time.time()

    return dualFishEyeImage, time1, dualFishEyeSalMaps

def change_salmap_color(salMaps, setupper):
    """
    グレースケールの顕著性画像を赤青のヒートマップに変換する。
    salMaps : グレースケール顕著性画像のndarray。(縦, 横)
    return : 　ヒートマップ化した顕著性画像のndarray(RGB)。(縦, 横, 3)
    """

    color_salMaps = cv2.applyColorMap((salMaps).astype(np.uint8), cv2.COLORMAP_JET)
    color_salMaps[setupper.mask] = [0,0,0]  #maskに記録してある顕著性マップではない部分は黒に戻す
    return color_salMaps

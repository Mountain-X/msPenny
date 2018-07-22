import cv2
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import sys


def get_f2e_npz(h, input_h):
    w = h*2

    od_xy_A = np.zeros((h,w,2)).astype(np.int)
    od_xy_B = np.zeros((h,w,2)).astype(np.int)

    y, x = np.meshgrid(np.arange(h),np.arange(w))
    v_t = -180 + 360*x/(w)
    v_p = 90 - 180*y/(h-1)
    od_xy_A[y,x,0], od_xy_A[y,x,1] = dualfish2equirectangular(v_p, v_t, input_h)

    v_t = 360*x/(w)
    od_xy_B[y,x,0], od_xy_B[y,x,1] = dualfish2equirectangular(v_p, v_t, input_h)

    np.savez('f2e_'+str(input_h)+'_'+str(h)+'.npz', A = od_xy_A, B = od_xy_B)


def dualfish2equirectangular(v_p, v_t, input_h):
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


def dualfish2equirectangular(v_p, v_t, input_h):
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


def f2e(np_image, prm):
    h = prm.f2e_size
    w = h*2
    input_h = np_image.shape[0]
    input_w = np_image.shape[1]
    frame_A = np_image[0:input_h, 0:input_h]
    frame_B = np_image[0:int(input_w/2), int(input_w/2):input_w]

    if not(os.path.isfile('f2e_'+str(input_h)+'_'+str(h)+'.npz')):
        get_f2e_npz(h, input_h)
    od_xy_load = np.load('f2e_'+str(input_h)+'_'+str(h)+'.npz')
    od_xy_A = od_xy_load['A']
    od_xy_B = od_xy_load['B']
    odiA = np.zeros((h, w, 3))
    odiA[0:h, 0:w] = frame_A[od_xy_A[0:h, 0:w, 0], od_xy_A[0:h, 0:w, 1]]
    odiB = np.zeros((h, w, 3))
    odiB[0:h, 0:w] = frame_B[od_xy_B[0:h, 0:w, 0], od_xy_B[0:h, 0:w, 1]]
    return odiA, odiB


class Setup_extract_omni_image():

    def __init__(self, extract_num = 6, extract_outputsize = 400,
                 f2e_size = 1000, view_angle = 90, elevation_angle = 45):
        self.extract_num = extract_num
        self.extract_outputsize = extract_outputsize
        self.f2e_size = f2e_size
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

    def get_angles(self):
        angles = np.empty((self.extract_num, self.extract_outputsize, self.extract_outputsize, 2))
        h = self.f2e_size
        w = h*2
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


def extract_omni_image(np_image, setupper):
    # ちょっとだけ改良
    omni_im = np.empty((setupper.extract_num, setupper.extract_outputsize,
                        setupper.extract_outputsize, 3)).astype(np.uint8)

    odiA, odiB = f2e(np_image, setupper)
    imcA = OmniImage(odiA)
    for num in range(len(setupper.cpA)):
        omni_im[num] = imcA.extract(setupper.cpA[num]).astype(np.uint8)
    imcB = OmniImage(odiB)
    for num in range(len(setupper.cpB)):
        omni_im[len(setupper.cpA) + num] = imcB.extract(setupper.cpB[num]).astype(np.uint8)

    """
    odiA, odiB = f2e(np_image, setupper)
    imcB = OmniImage(odiB)
    impB = np.array([imcB.extract(setupper.cpB[num]).astype(np.uint8) for num in range(len(setupper.cpB)) ])
    # impB = np.array(impB)
    # impB = np.uint8(impB)

    imcA = OmniImage(odiA)
    impA = np.array([imcA.extract(setupper.cpA[num]).astype(np.uint8) for num in range(len(setupper.cpA)) ])
    # impA = np.array(impA)
    # impA = np.uint8(impA)

    omni_im = np.empty((setupper.extract_num+2, setupper.extract_outputsize,
                        setupper.extract_outputsize, 3)).astype(np.uint8)

    omni_im[0] = impA[0]
    omni_im[1] = impA[1]
    omni_im[2] = impA[2]
    omni_im[3] = impB[0]
    omni_im[4] = impB[1]
    omni_im[5] = impB[2]
    """
    return omni_im

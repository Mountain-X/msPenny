import falcon
import json
from tqdm import tqdm
import numpy as np
import sys
from wsgiref import simple_server
import time
import cv2
import predictFromImage
from PIL import Image
import os

path2snapshot = '/home/tatsuyasuzuki/mspenny_saliency/pytorch-saliency-map/DenseSal/resume/train_jan11_201824/checkpoint.pth.tar'
outdir = '/home/tatsuyasuzuki/sal_server/static/'

# with Image.open('./planar0.png') as f:
#     img = np.array(f)
# imgs = [img]

# getPix2Angle  array
""" insert setup process"""
omniSetupper = predictFromImage.Setup_extract_omni_image(elevation_angle = 30)
maps = omniSetupper.get_angles()

# saliency map predector setup
salPredictor = predictFromImage.saliencyPredictor(path2snapshot, no_center_bias=True)

# human detection
#hdetector = HumanDetector()
#setThreeK = False


class ReturnSalMapAPI:

    def on_get(self, req, resp):
        resp.body = json.dumps(self.items)

    def on_put(self, req, resp):
        # humanDetected = False
        body = req.stream.read()
        base = time.time()
        # omniData = np.array(json.load(body)['image']['0'])[:, :, ::-1]
        omniData = np.frombuffer(body, dtype=np.uint8).reshape((736, 1472, 3))[:, :, ::-1]
        #omniData = np.frombuffer(body, dtype=np.uint8).reshape((1504, 3008, 3))[:, :, ::-1]
        angles = 'None'
        # Divide the omni pictures into multiple frames
        frames = predictFromImage.extract_omni_image(omniData, omniSetupper)
        salMaps = salPredictor.predict(images=frames)
        angle = salPredictor.getHightSaliencyDirection(salMaps)

        print('time of prediction:', time.time() - base, 's')
        print('angle :', angle)

        resp.body = json.dumps(angle)
        # path2img = os.path.join(outdir, "output.png")
        # cv2.imwrite("output.png", omniData)

        salMaps_max = np.zeros(len(salMaps))


        for i in range(len(salMaps)):
            salMaps_max[i] = salMaps[i].max()
        salMap_max = salMaps_max.max()

        for i in range(len(salMaps)):
            pil_im = Image.fromarray(np.uint8(salMaps[i]*255/salMap_max))
            salMaps[i] = np.array(pil_im.resize((400, 400)))


        time0 = time.time()
        dualFishEyeImage, time1, dualFishEyeSalMap = predictFromImage.inverse_extract_omni_image(salMaps, omniData, omniSetupper, overlaid_weight1 = 1, overlaid_weight2 = 0.7)
        print("time of inv_extraction :", time1-time0, 's')
        path2img = os.path.join(outdir, "output.png")
        cv2.imwrite(path2img, dualFishEyeImage)


def main():
    print("SERVER STARTED")
    api = falcon.API()
    api.add_route('/returnAngles', ReturnSalMapAPI())
    httpd = simple_server.make_server('192.168.11.61', 10026, api)
    httpd.serve_forever()


if __name__ == '__main__':
    main()

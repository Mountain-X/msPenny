import falcon
import json
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")
import sys
sys.path.append('/Users/reo911gt3/Desktop/mspenny/')
sys.path.append('/Users/reo911gt3/Desktop/mspenny/modules')
sys.path.append('/Users/reo911gt3/Desktop/mspenny/modules/yolo')
sys.path.append('/Users/reo911gt3/Desktop/mspenny/modules/yolo/data')
import modules
from modules.detectHuman import HumanDetector
import modules.omniPictureClass as omni
from modules.yolo import api
from wsgiref import simple_server
import time
import cv2

# getPix2Angle  array
""" insert setup process"""
omniSetupper = omni.Setup_extract_omni_image()
maps = omniSetupper.get_angles()

# human detection
hdetector = HumanDetector()
setThreeK = False


class ReturnAngleOfPersonAPI:

    def on_get(self, req, resp):
        resp.body = json.dumps(self.items)

    def on_put(self, req, resp):
        # humanDetected = False
        body = req.stream.read()
        # omniData = np.array(json.load(body)['image']['0'])[:, :, ::-1]
        omniData = np.frombuffer(body, dtype=np.uint8).reshape((736, 1472, 3))[:, :, ::-1]
        angles = 'None'
        # Divide the omni pictures into multiple frames
        frames = omni.extract_omni_image(omniData, omniSetupper)
        base = time.time()
        img_num, c1, c2, max_score = api.detect(frames)
        if img_num is not None:
            center_x = int(c2[0] - c1[0])
            center_y = int(c2[1] - c1[1])
            print(img_num, center_y, center_x, max_score)
            angles = maps[img_num][center_y, center_x][0]
        print("angle", angles)
        print(time.time() - base)

        for i, frame in enumerate(frames):
            imgName = str(i).zfill(4) + ".png"
            cv2.imwrite(imgName, frame[:, :, ::-1])
        """
        for i, frame in enumerate(tqdm(frames)):
            print("Searching for human presence")
            picked = hdetector.detecthuman(frame)
            if picked is not None:
                print("Human is found")
                humanDetected = True
                break
        if humanDetected:
            # if detected, caclulate the exact angle of where the person is
            xA, yA, xB, yB = picked
            cordOfDetectedHuman_x = xB - xA
            cordOfDetectedHuman_y = yB - yA

            # save the detected frame
            # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            # imsave("humanDetected.png", frame)

            angles = maps[i][cordOfDetectedHuman_y, cordOfDetectedHuman_x][0]
            print("angle", angles)
            """
        resp.body = json.dumps(angles)


def main():
    print("SERVER STARTED")
    api = falcon.API()
    api.add_route('/returnAngles', ReturnAngleOfPersonAPI())
    httpd = simple_server.make_server('192.168.11.14', 8000, api)
    httpd.serve_forever()


if __name__ == '__main__':
    main()

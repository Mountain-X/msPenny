import cv2
import json
import setupWheels

from modules.ultrasonic_sensor import ultrasonicSensor

import http.client
import traceback
import json


# calibrate wheels
controller = setupWheels.setup()
# controller.calibrateRotation()
controller.timeToMoveAnAngle = 0.009649194611443413

setThreeK = False

# initiate sensor
TRIG_PIN = 14
ECHO_PIN = 15
T = 23
distanceCalculator = ultrasonicSensor.UltraSonicSensor(TRIG_PIN, ECHO_PIN, T)

humanDetected = False


svr = '192.168.11.14:8000'
h = None
put_data = {"image":{}}


""" Connect the client to a server """


def connect():
    global h

    if h is not None:
        return
    h = http.client.HTTPConnection(svr)


def close():
    global h
    if h is None:
        return
    h.close()
    h = None


def put(image_0):
    global put_data
    connect()
    # put_data['image']['0'] = image_0.tolist()
    put_data['image']['0'] = image_0
    h.request('PUT', '/returnAngles', json.dumps(put_data))
    try:
        response = h.getresponse()
        body = response.read()
        angles = json.loads(body.decode('utf-8'))
        print("angles", angles)
        return angles
    except:
        traceback.print_exc()
    close()


while humanDetected is False:
    # get omni direction picture
    video_capture = cv2.VideoCapture(0)
    if setThreeK:
        video_capture.set(3, 3008)
        video_capture.set(4, 1504)
    ret, frame = video_capture.read()
    print("omni picture taken : ", ret)
    if ret:
        omniPic = frame[:, :, ::-1]
        angles = put(omniPic)
        if angles is not None:
            humanDetected = True
    video_capture.release()

# rotate untill the person is in the front
controller.moveSomeAngles(angles)

# move forward untill the ultrasensor return dist < 30
while True:
    controller.forward()
    dist = distanceCalculator.calcDistance()
    print("distance is {} cm".format(dist))
    if dist < 20:
        controller.stop()
        break

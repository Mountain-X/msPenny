# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


class HumanDetector(object):
    def __init__(self, loadFromFile=False):
        self.loadFromFile = loadFromFile
        # construct the argument parse and parse the arguments
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detecthuman(self, image):
        # loop over the image paths
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        if self.loadFromFile:
            image = cv2.imread(image)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4),
                                                     padding=(8, 8),
                                                     scale=1.05)
        # rects = [[...]]
        # print(rects)
        if len(rects) == 0:
            return None

        """
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        """

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        picked = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        """
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        """
        print("picked", picked)
        maxRect = picked[0]
        print("maxRect", maxRect)
        if len(picked) > 1:
            print("multiple person found")
            maxRect = picked[0]
            maxRectSize = (maxRect[2]-maxRect[0]) * (maxRect[1] - maxRect[0])
            for i in range(len(picked) - 1):
                cmpRect = picked[i+1]
                cmpRectSize = (cmpRect[2]-cmpRect[0]) * (cmpRect[1] - cmpRect[0])
                if cmpRectSize > maxRectSize:
                    maxRect = picked[i+1] 
                    maxRectSize = (maxRect[2]-maxRect[0]) * (maxRect[1] - maxRect[0])
        return maxRect


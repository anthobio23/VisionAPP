#! /usr/bin/python3.9

import numpy as np
from cv2 import cv2 as cv

def SortPoints(points):

    n_points = np.concatenate([points[0], points[1],
        points[2], points[3]]).tolist()

    y_order = sorted(n_points, 
            key=lambda n_points: n_points[1])
    
    x1_order = y_order[:2]
    x2_order = sorted(x1_order, 
            key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, 
            key=lambda x2_order: x2_order[0])
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alignment(image, width, heigth):

    image_alignment = None
    Grays = cv.cvtColor(image, 
            cv.COLOR_BGR2GRAY)

    TypeThreshold, Threshold = cv.threshold(Grays, 150, 155, 
            cv.THRESH_BINARY)
    cv.imshow("Threshold", Threshold)

    contours = cv.findContours(Threshold,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)[0]

    contours = sorted(contours, key=cv.contourArea,
            reverse=True)[:1]

    for c in contours:

        Epsilon = 0.01 * cv.arcLength(c, True)
        Approx = cv.approxPolyDP(c, 
                Epsilon,
                True)

        if len(Approx) == 4:

            points = SortPoints(Approx)
            points_1 = np.float32(points)
            points_2 = np.float32(
                    [0, 0], 
                    [width, 0], 
                    [0, heigth],
                    [width, heigth])
            M = cv.getPerspectiveTransform(
                    points_1,
                    points_2)
            image_alignment = cv.warpPerspective(image,
                    M, (width))

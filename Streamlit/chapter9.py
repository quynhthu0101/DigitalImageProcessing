import cv2
import numpy as np
L = 256

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    imgout = cv2.erode(imgin, w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout

def Contour(imgin):
    # bắt buộc imgin là ảnh nhị phân (có 2 màu: đen 0 và trắng 255)
    M, N = imgin.shape
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours,_ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = contours[0]
    n = len(contours)
    for i in range (0, n - 1):
        x1 = contours[i][0][0]
        y1 = contours[i][0][1]

        x2 = contours[i + 1][0][0]
        y2 = contours[i + 1][0][1]

        cv2.line(imgout, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1 = contours[n - 1][0][0]
    y1 = contours[n - 1][0][1]

    x2 = contours[0][0][0]
    y2 = contours[0][0][1]

    cv2.line(imgout, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return imgout


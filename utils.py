import cv2 as cv
import numpy as np


def draw_two_lines(img):
    img = cv.line(img=img, pt1=(40, 250), pt2=(600, 250), color=(0, 255, 0), thickness=2, lineType=cv.LINE_8, shift=0)
    img = cv.line(img=img, pt1=(40, 440), pt2=(600, 440), color=(0, 255, 0), thickness=2, lineType=cv.LINE_8, shift=0)

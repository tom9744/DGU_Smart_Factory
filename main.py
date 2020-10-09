import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils

""" Read Camera """
camera = cv.VideoCapture(1)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

""" Read Template """
template = cv.imread("resources/template_2.png", 0)
template = cv.resize(template, (25, 220))
height, width = template.shape  # 템플릿 이미지 세로, 가로 길이

MORPH_KERNEL = np.ones((1, 1), np.uint8)  # 모폴로지 변형 Structure Element Size

while True:
    ret, img = camera.read()
    if not ret:
        print("Error, There's no Camera")
        break

    """ Draw lines on an image """
    # utils.draw_two_lines(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 영상으로 변환

    result = cv.matchTemplate(gray, template, cv.TM_SQDIFF_NORMED)

    # minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    # x, y = minLoc

    # print(x, y, h, w)
    # img = cv.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=1)

    res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res > threshold)
    counter = 0
    for pt in zip(*loc[::-1]):
        counter = counter + 1
        # cv.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)

    print(counter)
    cv.imshow("Input", img)
    cv.imshow("Template", template)

    # reverse = cv.bitwise_not(gaussian)
    # cv.imshow("Gaussian", reverse)
    #
    # # opened = cv.morphologyEx(reverse, cv.MORPH_OPEN, MORPH_OPENING_KERNEL)
    # dilated = cv.dilate(gaussian, MORPH_KERNEL, iterations=1)
    # cv.imshow("Opening", dilated)


    key = cv.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

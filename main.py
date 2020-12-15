import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


""" Read Camera """
camera = cv.VideoCapture(0)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

""" 모폴로지 변형 Structure Element """
MORPH_KERNEL = np.ones((3, 3), np.uint8)  # Size(1,1)
MORPH_ERODE_KERNEL = np.ones((1, 1), np.uint8)  # Size(1,1)
MORPH_DILATE_KERNEL = np.ones((1, 1), np.uint8)  # Size(1,1)

while True:
    ret, image = camera.read()
    if not ret:
        print("Error, There's no Camera")
        break

    """ ROI(Region of Interest) 설정 """
    cropped_img = image[270: 270 + 80, 150: 170 + 320]  # Y축 시작 - 종료 / X축 시작 - 종료

    gray_scale = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)  # 그레이스케일 영상으로 변환

    ret, binary = cv.threshold(gray_scale, 100, 255, cv.THRESH_BINARY)  # 75~255 밝기의 픽셀 255로 이진화

    reversed_binary = ~binary  # 이미지 반전

    dilated = cv.dilate(reversed_binary, MORPH_DILATE_KERNEL, iterations=3)  # 모폴로지 Dilate 연산으로 끊어진 픽셀 연결

    """ Closing -> Dilation """
    for _ in range(3):
        opening = cv.morphologyEx(dilated, cv.MORPH_OPEN, MORPH_KERNEL)

    """ Find contours in an image """
    contours, hierarchy = cv.findContours(opening, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    """ Count contours that are bigger than 90 (Average size 100)"""
    count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)

        if area > 80:
            count = count + 1

    # FOR DEBUGGING
    # print(count)
    # cv.imshow("Original", image)
    # cv.imshow("Gray", gray_scale)
    # cv.imshow("Binary", binary)
    # cv.imshow("Reversed Binary", reversed_binary)
    # cv.imshow("Reversed Binary Open", opening)

    """ 'Q' 입력이 들어오면, 반복문 종료 """
    key = cv.waitKey(1)
    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

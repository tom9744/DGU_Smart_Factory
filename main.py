import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils


def template_match(input):
    """ Read Template """
    template = cv.imread("resources/template_2.png", 0)
    template = cv.resize(template, (25, 220))
    height, width = template.shape  # 템플릿 이미지 세로, 가로 길이

    cv.imshow("Template", template)

    res = cv.matchTemplate(input, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res > threshold)
    counter = 0
    for pt in zip(*loc[::-1]):
        counter = counter + 1
        cv.rectangle(input, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)

    print(counter)


""" Read Camera """
camera = cv.VideoCapture(1)
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
    cropped_img = image[210: 210 + 100, 165: 165 + 250]  # Y축 시작 - 종료 / X축 시작 - 종료

    gray_scale = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)  # 그레이스케일 영상으로 변환

    ret, binary = cv.threshold(gray_scale, 75, 255, cv.THRESH_BINARY)  # 75~255 밝기의 픽셀 255로 이진화

    reversed_binary = ~binary  # 이미지 반전

    dilated = cv.dilate(reversed_binary, MORPH_DILATE_KERNEL, iterations=3)  # 모폴로지 Dilate 연산으로 끊어진 픽셀 연결

    """ Closing -> Dilation """
    for _ in range(3):
        opening = cv.morphologyEx(dilated, cv.MORPH_OPEN, MORPH_KERNEL)

    cv.imshow("Original", gray_scale)
    cv.imshow("Binary", binary)
    cv.imshow("Reversed Binary", reversed_binary)
    cv.imshow("Reversed Binary Open", opening)

    """ 'Q' 입력이 들어오면, 반복문 종료 """
    key = cv.waitKey(1)
    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

# blurred = cv.GaussianBlur(gray_scale, (5, 5), 0)    # 그레이스케일에 가우시안 블러 적용 (노이즈 제거)
# sobel_x = cv.Sobel(blurred, cv.CV_8U, 1, 0, ksize=3)
# sobel_y = cv.Sobel(blurred, cv.CV_8U, 0, 1, ksize=3)
# laplacian = cv.Laplacian(blurred, cv.CV_8U, ksize=5)
#
# erosion = cv.erode(laplacian, MORPH_KERNEL, iterations=1)
#
# cv.imshow("Blurred Image", blurred)
# cv.imshow("Blurred Image / SobelX", sobel_x)
# cv.imshow("Blurred Image / SobelY", sobel_y)
# cv.imshow("Sobel X + Y", sobel_x + sobel_y)
# cv.imshow("Blurred Image / Laplacian", laplacian)
# cv.imshow("Blurred Image / Erosion", erosion)

import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0, cv.CAP_DSHOW)

MORPH_KERNEL = np.ones((1, 1), np.uint8)  # 모폴로지 변형 Structure Element Size

while True:
    ret, img = camera.read()
    gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환

    gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 5)
    cv.imshow("Gaussian", gaussian)

    reverse = cv.bitwise_not(gaussian)
    cv.imshow("Gaussian", reverse)

    # opened = cv.morphologyEx(reverse, cv.MORPH_OPEN, MORPH_OPENING_KERNEL)
    dilated = cv.dilate(gaussian, MORPH_KERNEL, iterations=1)
    cv.imshow("Opening", dilated)

    key = cv.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

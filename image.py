import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = cv2.imread('resources/source2.png')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('resources/lighter.png', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template,cv2.TM_CCOEFF_NORMED)
threshold = 0.845333
loc = np.where(res > threshold)
print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 2)

cv2.imwrite('result.png', img_rgb)
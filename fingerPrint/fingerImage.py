import os
import cv2
import dlib
import numpy as np
import matplotlib as mat

image = cv2.imread("Test/finger.png")

#contrast and brightness to 0.8 and 25
brightness = 25
contrast = 0.8
img = np.int16(image)
img = img * (contrast/127+1) - contrast + brightness
img = np.clip(img, 0, 255)
img = np.uint8(img)

#Gray
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cImg = clahe.apply(gray_image)

equ = cv2.equalizeHist(cImg)

#Normalized
norm_image = cv2.normalize(equ, None, 0, 255 , cv2.NORM_MINMAX)

#adaptive (gaussian c) threshold for block size of 15 and constant 2
thresholdImg = cv2.adaptiveThreshold(norm_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

#Gaussian Blur
thImg = cv2.blur(thresholdImg, (5,5))
thImg2 = cv2.medianBlur(thImg,5)
# 閾值
smoothImg = cv2.adaptiveThreshold(thImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
smoothImg2 = cv2.adaptiveThreshold(thImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)



cv2.imwrite('gray_image.png',gray_image)
cv2.imshow('og_image',img)
cv2.imshow('gray_image',smoothImg2)
cv2.imshow('test',smoothImg)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()

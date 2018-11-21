import cv2
import numpy as np

cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
img1 = cv2.imread('C:\Python Stuff\Entry1.jpg')


grayscaled1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


retval1,threshold1 = cv2.threshold(grayscaled1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#cv2.resizeWindow('img1', 600,600)

#cv2.imshow('original1',img1)

cv2.imshow('Output',threshold1)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('C:\Python Stuff\Entry1.jpg',0)
edges = cv2.Canny(img,0,75)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
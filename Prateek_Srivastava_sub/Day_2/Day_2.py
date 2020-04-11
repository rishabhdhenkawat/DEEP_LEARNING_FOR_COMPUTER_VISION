# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:09:55 2020

@author: sripr
"""

from skimage import exposure
from skimage import feature
import cv2
path=r"C:\Users\sripr\Desktop\DEEP_LEARNING_FOR_COMPUTER_VISION\Prateek_Srivastava_sub\Day 2\cat.png"
logo=cv2.imread(path)
(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(2,2),
cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
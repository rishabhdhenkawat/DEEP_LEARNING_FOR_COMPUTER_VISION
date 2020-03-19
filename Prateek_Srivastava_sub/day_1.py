# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:00:36 2020

@author: sripr
"""

from PIL import Image
import numpy as np
img_w, img_h = 200, 200
data = np.zeros((img_h, img_w,3), dtype=np.uint8)
data[40:80,40:80,:] = 255*(np.ones((40,40,3),dtype=np.uint8))  #left eye
data[40:80,120:160,:]=255*(np.ones((40,40,3),dtype=np.uint8)) #Right eye
data[120:140,60:140,:]=255*(np.ones((20,80,3),dtype=np.uint8))#Mouth

img = Image.fromarray(data, 'RGB')
img.save('test.png')
img.show()
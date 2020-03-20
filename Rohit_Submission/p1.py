# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:26:48 2020

@author: Lenovo
"""

from PIL import Image
import numpy as np
img_w, img_h = 300, 300
data = np.zeros((img_h, img_w, 3), dtype=np.uint8)


for i in range(60,101):#for eyes
  for j in range(70,111):
    data[i,j]=[255,255,255]
    data[i,j+120]=[255, 255, 255]
    
for i in range(5,16):#for white border
  for j in range(5,296):
    data[i,j]=[255,255,255]
    data[i+280,j]=[255,255,255]
    data[j,i]=[255,255,255]
    data[j,i+280]=[255,255,255]
    
for i in range(150,171):#for nose
  for j in range(140,161):
    data[i,j]=[255,255,255]
    
for i in range(230,251):#for mouth
  for j in range(100,201):
    data[i,j]=[255, 255, 255]
    
img = Image.fromarray(data, 'RGB')
img.save('test.png')
img.show()

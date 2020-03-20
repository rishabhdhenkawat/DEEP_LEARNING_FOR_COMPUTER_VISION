# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 03:44:08 2020

@author: nomaa
"""

import cv2
import numpy as np

#creating a dark face using numbers
im_w=300
im_h=300
pixel_data=np.zeros((im_h,im_w,3),dtype=np.uint8)

#dimensions of eyes and mouth
eye_wid=40
mouth_wid=100
mouth_ht=26

#giving it eyes
for i in range(1,eye_wid+1):
    for j in range(1,eye_wid+1):
        pixel_data[100-int(eye_wid/2)+i][100-int(eye_wid/2)+j]=255
        pixel_data[100-int(eye_wid/2)+i][200-int(eye_wid/2)+j]=255
        
#giving it mouth
for i in range(1,mouth_ht+1):
    for j in range(1,mouth_wid+1):
        pixel_data[230-int(mouth_ht/2)+i][150-int(mouth_wid/2)+j]=255
        
#printing the image
cv2.imshow('image',pixel_data)
cv2.waitKey(1000)
cv2.destroyAllWindows()  

  
#how about giving it some eyebrows
for i in range(1,5):
    for j in range(1,61):
        pixel_data[100-int(eye_wid/2)-9+i][100-int(eye_wid/2)-10+j]=255
        pixel_data[100-int(eye_wid/2)-9+i][200-int(eye_wid/2)-10+j]=255
        
#printing the image
cv2.imshow('image',pixel_data)
cv2.waitKey(5000)
cv2.destroyAllWindows()    
        
            
    


        
    
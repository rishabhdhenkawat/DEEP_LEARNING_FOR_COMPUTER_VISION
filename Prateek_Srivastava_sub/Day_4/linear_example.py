# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:52:54 2020

@author: sripr
"""

import numpy as np
import cv2

labels=["dog","cat","panda"]
np.random.seed(1) #For pseudo Random Numbers

W=np.random.randn(3,3072)
b=np.random.randn(3)

#loading
path=r"C:\Users\sripr\Desktop\DEEP_LEARNING_FOR_COMPUTER_VISION\Prateek_Srivastava_sub\Day_4\beagle.png"
orig=cv2.imread(path)
image=cv2.resize(orig,(32,32)).flatten()
#converting (32,32,3) array into 3072 vector

scores=W.dot(image)+b

for (label,score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))

#Draw label with highest scoreon image as our prediction
    
cv2.putText(orig,"Label: {}".format(labels[np.argmax(scores)]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

#Display Input image
cv2.imshow("image",orig)
cv2.waitKey(0)
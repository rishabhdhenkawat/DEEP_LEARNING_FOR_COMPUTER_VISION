import numpy as np
import matplotlib.pyplot as plt

h,w = 200,200
img = np.zeros((h,w,3),dtype= np.uint8)
plt.imshow(img)
eyes = 255*(np.ones((35,40,3),dtype= np.uint8))
img[25:60,25:65,:]= eyes
img[25:60,135:175,:]= eyes
mouth =255*(np.ones((15,70,3),dtype= np.uint8))
img[150:165,65:135,:]=mouth
plt.imshow(img)


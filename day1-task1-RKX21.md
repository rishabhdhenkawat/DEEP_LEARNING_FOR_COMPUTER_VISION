from PIL import Image
import numpy as np
img_w, img_h = 300, 300
data = np.zeros((img_h, img_w,3), dtype=np.uint8)

data[50:100,50:100,:] = 255*(np.ones((50,50,3),dtype=np.uint8))  #left eye

data[50:100,200:250,:]=255*(np.ones((50,50,3),dtype=np.uint8)) #Right eye

data[200:225,100:200,:]=255*(np.ones((25,100,3),dtype=np.uint8))#Mouth

img = Image.fromarray(data, 'RGB')
img.save('test.png')
img.show()

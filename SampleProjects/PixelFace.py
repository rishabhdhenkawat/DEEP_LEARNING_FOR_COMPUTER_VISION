from PIL import Image
import numpy as np
white = [255,255,255]

#for static case only
img_w, img_h = 200, 200
data = np.zeros((img_h, img_w,3), dtype=np.uint8)

#Making the Eyes
for i in range(35,65):
    for j in range(35,65):
        data[i,j] = white
        data[i,j + 100] = white

#Making the Lips
for i in range(140,160):
    for j in range(50,150):
        data[i,j] = white

img = Image.fromarray(data, 'RGB')
img.save('test.png')
img.show()


from PIL import Image
import numpy as np
img_w, img_h = 500, 500
data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
data[125:200, 125:200] = [255, 255, 255] 
data[125:200, 300:375] = [255, 255, 255] 
data[350:375, 175:325] = [255, 255, 255] 
img = Image.fromarray(data, 'RGB')
img.save('test.png')
img.show()


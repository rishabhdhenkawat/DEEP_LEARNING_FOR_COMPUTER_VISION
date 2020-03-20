from PIL import Image
import numpy as np
img_w, img_h = 300, 300
test = np.zeros((img_h, img_w, 3), dtype=np.uint8) 
test.show()
test[50:100,50:100,:] = 255 * np.ones((50,50,3), dtype=np.uint8)
test[50:100,200:250,:] = 255 * np.ones((50,50,3), dtype=np.uint8)
test[180:200,100:200,:] = 255 * np.ones((20,100,3), dtype=np.uint8)
img = Image.fromarray(test, 'RGB')
img.show()


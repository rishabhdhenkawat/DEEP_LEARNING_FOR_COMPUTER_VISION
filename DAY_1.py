from PIL import Image
import numpy as np
img_w, img_h = 1000, 1000
data = np.zeros((img_h, img_w, 3), dtype=np.uint8)



for i in range(100):
    for j in range(100):
        data[i+200 , j+300] = [25,66,78]
        

for k in range(100):
    for l in range(100):
        data[k+200 , l+700] = [245,58,63]
        
for m in range(100):
    for n in range(500):
        data[m+700 , n+250] = [56,85,35]
        
img = Image.fromarray(data, 'RGB')

img.show()

#looking a bit out of the given format but it looked awesome..........

from PIL import Image
import numpy as np


class Pic:
    
    # Used to initialise the basic array and dimensions
    def __init__(self,img_w= 400, img_h=400):
        self.img_w = img_w
        self.img_h = img_h
        self.pic_array = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Displays the image
    def show(self):
        self.img = Image.fromarray(self.pic_array, 'RGB')
        self.img.save('test.png')
        self.img.show()
        
    # takes co-ordinates of top-left and bottom-right corner
    # and draws into the array
    def white_rectangle(self,top_left,bottom_right):
        for i in range(top_left[0],bottom_right[0]):
            for j in range(top_left[1],bottom_right[1]):
                self.pic_array[i,j] = [255, 255, 255]
       
project = Pic()

## leaving a 50 pixel gap meaning 100 pixel gap in each co ordinate for a 50x50 square
project.white_rectangle((150,125),(200,175))
project.white_rectangle((150,225),(200,275))

## smile 
project.white_rectangle((250,150),(275,250))

#display
project.show()
from skimage import exposure
from skimage import feature
import cv2

#Opening Image
img=cv2.imread("/home/avishrant/cat.png")

#HOG evaluation
(H, himg) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True)

#Rescaling
himg = exposure.rescale_intensity(himg, out_range=(0, 255))
himg = himg.astype("uint8")

#Output
cv2.imshow("HOG Image", himg)
cv2.waitKey(0)
cv2.destroyAllWindows()

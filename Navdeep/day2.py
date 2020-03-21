from skimage import exposure
from skimage import feature
import cv2
logo=cv2.imread("cat_pic.jpg")
(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(16, 16),
cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

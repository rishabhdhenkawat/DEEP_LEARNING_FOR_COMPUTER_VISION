from skimage import exposure
from skimage import feature
import cv2
logo=cv2.imread("Capture.PNG")
(H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(4,4),
cells_per_block=(3,3), transform_sqrt=True, block_norm="L1",
visualize=True,multichannel=True)
hogImage = exposure.rescale_intensity(hogImage,in_range=(.2,.8), out_range=(0, 210))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
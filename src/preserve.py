# preserve color
import numpy as np
import cv2
from PIL import Image
from skimage import color

# given 2 input uint8 images (content and gallery) of same size
# returns the gallery image by preserving the color from content
# by keeping luminosity of LAB color space of gallery and original color
def preserve_lab(content, gallery):
	content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB)
	gallery_lab = cv2.cvtColor(gallery, cv2.COLOR_BGR2LAB)
	convert = content_lab
	convert[..., 0] = gallery_lab[..., 0]
	return np.array(cv2.cvtColor(convert, cv2.COLOR_LAB2BGR), 'uint8');
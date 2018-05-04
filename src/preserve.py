# preserve color
import numpy as np
from PIL import Image
from skimage import color

# given 2 input uint8 images (content and gallery) of same size
# returns the gallery image by preserving the color from content
# by keeping luminosity of LAB color space of gallery and original color
def preserve_lab(content, gallery):
	content_lab = color.rgb2lab(content)
	gallery_lab = color.rgb2lab(gallery)
	convert = content_lab
	convert[..., 0] = gallery_lab[..., 0]
	return np.array(color.lab2rgb(convert)*256, 'uint8');


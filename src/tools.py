import scipy.misc as sm
import numpy as np
import PIL

def load_image(path, given_size=None):
    image = PIL.Image.open(path)
    np_img = np.float32(image)

    # convert grayscale to 3-channel image
    if len(np_img.shape) < 3 or np_img.shape[2] != 3:
        np_img = np.stack((np_img,) * 3, -1)
        print("shape after converting:", np_img.shape)
    if given_size:
        np_img = sm.imresize(np_img, given_size)

    return np_img


def save_image(path, given_image):
    image = np.clip(given_image, 0, 255)
    image = image.astype(np.uint8)
    sm.imsave(path, image)

if __name__ == "__main__":
    # Test
    image = load_image("../images/input.jpg", (800, 600))
    save_image("../images/output.jpg", image)
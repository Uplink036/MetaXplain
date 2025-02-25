import numpy as np
import PIL
from PIL import Image

def np_to_pil(image):
    return Image.fromarray(image.astype('uint8'))

def pil_to_np(image):
    return np.array(image)

def rotate_PIL(image, angel):
    return image.rotate(angel)

def rotate(image: np.array, angle: int):
    shape = image.shape
    pil = np_to_pil(image)
    rotated = rotate_PIL(pil, angle)
    flat_matrix = pil_to_np(rotated)
    return flat_matrix.reshape(shape)
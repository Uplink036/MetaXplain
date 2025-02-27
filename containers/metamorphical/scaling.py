import numpy as np
from PIL import Image

def np_to_pil(image):
    return Image.fromarray(image.astype('uint8'))

def pil_to_np(image):
    return np.array(image)

def upscale(image, new_size=(256, 256)):
    shape = image.shape
    pil = np_to_pil(image)
    larger = pil.resize(new_size, resample=0)
    same = larger.resize(shape, resample=0)
    flat_matrix = pil_to_np(same)
    return flat_matrix.reshape(shape)


def downscale(image, new_size=(12, 12)):
    shape = image.shape
    pil = np_to_pil(image)
    smaller = pil.resize(new_size, resample=0)
    same = smaller.resize(shape, resample=0)
    flat_matrix = pil_to_np(same)
    return flat_matrix.reshape(shape)

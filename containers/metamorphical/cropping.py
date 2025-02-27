
from copy import deepcopy
import random
import numpy as np
from PIL import Image

def np_to_pil(image):
    return Image.fromarray(image.astype('uint8'))

def pil_to_np(image):
    return np.array(image)

def crop(image, direction = 0):
    shape = image.shape
    if direction == 0: #horizontal
        options = range(0, round(shape[0] / 2))
    elif direction == 1:
        options = range(0, round(shape[1] / 2))

    choice = random.choice(options) 
    new_image = deepcopy(image)
    if direction == 0: #horizontal
        if choice > shape[0] / 2:
            coloums_to_delete = np.arange(choice+round(3/4*shape[0]), shape[0])
        else:
            coloums_to_delete = np.arange(0, choice)
    elif direction == 1:
        options = range(0, shape[1] / 2)
        if choice > shape[0] / 2:
            coloums_to_delete = np.arange(choice+round(3/4*shape[0]), shape[0])
        else:
            coloums_to_delete = np.arange(0, choice)

    cropped_image =  np.delete(new_image, coloums_to_delete, axis=direction)
    pil = np_to_pil(cropped_image)
    resized = pil.resize(shape, resample=0)
    flat_matrix = pil_to_np(resized)
    return flat_matrix.reshape(shape)



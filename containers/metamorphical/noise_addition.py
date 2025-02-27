import numpy as np
from PIL import Image

def noisey(image):
    shape = image.shape
    image_mean = np.mean(image)
    image_std= np.std(image)
    random_mean = max(image_mean/4, 2)
    random_std= max(image_std/4, 2)
    noise = np.random.normal(random_mean, random_std, shape)
    return image + noise 
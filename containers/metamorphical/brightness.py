import numpy as np

def change_brightness(image, alpha=0.9):
    shape = image.shape
    new_image = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            new_image[x][y] = round(image[x][y]*alpha)
    return new_image
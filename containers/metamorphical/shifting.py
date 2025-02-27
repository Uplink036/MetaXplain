from math import cos, sin, pi
import numpy as np


def shift(image, angle=0, level=5):
    shape = image.shape
    new_image = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            rads = angle*pi/180
            new_x = x + round(level*cos(rads))
            new_y = y + round(level*sin(rads))

            if not (new_x >= 0 and new_x <shape[0]):
                continue
            if not (new_y >= 0 and new_y <shape[1]):
                continue

            new_image[new_x][new_y] = image[x][y]
    return new_image


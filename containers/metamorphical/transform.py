import random
import numpy as np
from data import loader
from rotation import rotate

def transform_rotate(data, dataset, image):
    for angle in range(-15, 15+5, 5):
        random_noise =  random.normalvariate(0, 4)
        rotated_image = rotate(image, angle+random_noise)
        meta_dict = {
            "label": data["label"],
            "size": "28x28",
            "image": rotated_image.flatten().tolist(),
            "status": "rotation",
            "angle": angle+random_noise
        }
        dataset.upload(meta_dict)

from noise_addition import noisey
def transform_noise(data, dataset, image):
    for times in range(0, 5):
        noisey_image = noisey(image)
        meta_dict = {
            "label": data["label"],
            "size": "28x28",
            "image": noisey_image.flatten().tolist(),
            "status": "noise",
            "iteration": times
        }
        dataset.upload(meta_dict)
        image = noisey_image

from shifting import shift
def transform_shift(data, dataset, image):
    for angle in range(0, 360, 15):
        random_noise =  random.normalvariate(0, 4)
        rotated_image = shift(image, angle+random_noise)
        meta_dict = {
            "label": data["label"],
            "size": "28x28",
            "image": rotated_image.flatten().tolist(),
            "status": "shift",
            "angle": angle+random_noise
        }
        dataset.upload(meta_dict)

from cropping import crop
def transform_crop(data, dataset, image):
    for times in range(0, 5):
        cropped_image = crop(image, times % 2)
        meta_dict = {
            "label": data["label"],
            "size": "28x28",
            "image": cropped_image.flatten().tolist(),
            "status": "crop",
            "direction": times % 2
        }
        dataset.upload(meta_dict)

from brightness import change_brightness
def transform_brightness(data, dataset, image):
    for level in range(1, 5):
        cropped_image = change_brightness(image,1-level*0.1)
        meta_dict = {
            "label": data["label"],
            "size": "28x28",
            "image": cropped_image.flatten().tolist(),
            "status": "brightness",
            "level": 1-level*0.1
        }
        dataset.upload(meta_dict)

if __name__ == "__main__":
    dataset = loader()

    for data in dataset.find({"status":"test"}):
        image = data["image"]
        np_image = np.reshape(image, (28, 28))
        transform_rotate(data, dataset, np_image)
        transform_noise(data, dataset, np_image)
        transform_shift(data, dataset, np_image)
        transform_crop(data, dataset, np_image)
        transform_brightness(data, dataset, np_image)
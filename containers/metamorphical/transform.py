import random
import numpy as np
from data import loader
from rotation import rotate

if __name__ == "__main__":
    dataset = loader()
    
    for data in dataset.find({"status":"test"}):
        image = data["image"]
        np_image = np.reshape(image, (28, 28))
        for angle in (-45, 45, 15):
            random_noise =  random.normalvariate(0, 7)
            rotated_image = rotate(image, angle+random_noise)
            meta_dict = {
                "label": data["label"],
                "size": "28x28",
                "image": rotated_image.flatten(),
                "status": "rotation"
            }
            dataset.upload(meta_dict)
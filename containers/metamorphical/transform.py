import random
import numpy as np
from data import loader
from rotation import rotate

if __name__ == "__main__":
    dataset = loader()
    
    for data in dataset.find({"status":"test"}):
        image = data["image"]
        np_image = np.reshape(image, (28, 28))
        for angle in range(-15, 15+5, 5):
            random_noise =  random.normalvariate(0, 4)
            rotated_image = rotate(np_image, angle+random_noise)
            meta_dict = {
                "label": data["label"],
                "size": "28x28",
                "image": rotated_image.flatten().tolist(),
                "status": "rotation",
                "angle": angle+random_noise
            }
            dataset.upload(meta_dict)
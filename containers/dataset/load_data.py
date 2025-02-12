import numpy as np # linear algebra
import struct
from dotenv import load_dotenv
import pymongo
import os
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = image_data[i * rows * cols:(i + 1) * rows * cols]
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        print("Loading Data...")
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        print("Sending to database...")
        try:
            database_url = os.environ.get("DATABASE_URL")
            client = pymongo.MongoClient(database_url)
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
            print("Are you sure your database is on and this can reach it?")

        db = client["MNIST"]
        clean_db = int(os.getenv('SCRUB_DB'))
        if clean_db == 1:
                client.drop_database("dataset")
        db = client["dataset"]
        dataset = db["dataset"]
        dataset.insert_many([{"image": img, "size": "28x28", "label": label} 
                             for img, label in zip(x_train, y_train)])
        dataset.insert_many([{"image": img, "size": "28x28", "label": label} 
                             for img, label in zip(x_test, y_test)])
        print(f"Database has {dataset.count_documents({})} entires")
        
if __name__ == "__main__":
    print("Starting...")
    dataloader = MnistDataloader("train-images.idx3-ubyte",
                    "train-labels.idx1-ubyte",
                    "t10k-images.idx3-ubyte",
                    "t10k-labels.idx1-ubyte")
    dataloader.load_data()
    print("Ending...")
import os 
import pymongo

class batchLoader():
    def __init__(self):
        self.batch_nr = 0
        self.batch_size = 32
        self.client = None
        self.database = None
        self.collection = None
        self._init_db()

    def _init_db(self):
        try:
            database_url = os.environ.get("DATABASE_URL")
            client = pymongo.MongoClient(database_url)
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
            print("Are you sure your database is on and this can reach it?")
            raise ConnectionError
        self.client = client
        self.database = client["MNIST"]
        self.collection = self.database["dataset"]

    def exit(self):
        self.client.close()

    def _batch_find(self):
        data = self.collection.find({}).skip(self.batch_nr*self.batch_size).limit(self.batch_size)
        self.batch_nr += 1
        return data
    
    def batch(self):
        batch_data = self._batch_find()
        labels = [0]*self.batch_size
        inputs = [[]]*self.batch_size
        for index, batch in enumerate(batch_data, 0):
            labels[index ]= batch["label"]
            inputs[index] = batch["image"]   
        return labels, inputs
    
    def reset(self):
        self.batch_nr = 0

    def get_number_of_batches(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        return round(self.collection.count_documents({}) / batch_size)

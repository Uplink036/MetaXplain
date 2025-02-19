import os 
import pymongo
import dotenv

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
        
    def batch(self):
        data = self.collection.find({"skip":self.batch_nr*self.batch_size,
                                     "limit":self.batch_size})
        labels = [x["label"] for x in data]
        data = [x["image"] for x in data]
        return labels, data

    def exit(self):
        pass

    def get_number_of_batches(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        return self.collection.count_documents({}) / batch_size
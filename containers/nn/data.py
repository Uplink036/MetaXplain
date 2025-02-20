import os 
import pymongo
import logging
logger = logging.getLogger(__name__)


class batchLoader():
    def __init__(self):
        logger.info("Creating BatchLoader")
        self.batch_nr = 0
        self.batch_size = 32
        self.client = None
        self.database = None
        self.collection = None
        self._init_db()

    def _init_db(self):
        logger.info("Connecting to database")
        try:
            database_url = os.environ.get("DATABASE_URL")
            client = pymongo.MongoClient(database_url)
            client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            logger.fatal(err)
            logger.fatal("Are you sure your database is on and this can reach it?")
            raise ConnectionError
        self.client = client
        self.database = client["MNIST"]
        self.collection = self.database["dataset"]

    def exit(self):
        logger.info("Closing connection to database")
        self.client.close()

    def _batch_find(self, query = {}):
        
        data = self.collection.find(query).skip(self.batch_nr*self.batch_size).limit(self.batch_size)
        self.batch_nr += 1
        return data
    
    def batch(self, query={}):
        batch_data = self._batch_find(query)
        labels = [0]*self.batch_size
        inputs = [[]]*self.batch_size
        for index, batch in enumerate(batch_data, 0):
            labels[index ]= batch["label"]
            inputs[index] = batch["image"]   
        return labels, inputs
    
    def reset(self):
        logger.info("Reseting to start")
        self.batch_nr = 0

    def get_number_of_batches(self, batch_size = None, query={}):
        if batch_size is None:
            batch_size = self.batch_size

        return round(self.collection.count_documents(query) / batch_size)

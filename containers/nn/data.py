import os 
import pymongo
import logging

import pymongo.collation
import pymongo.collection
logger = logging.getLogger(__name__)


class batchLoader():
    def __init__(self, collection_name):
        logger.info("Creating BatchLoader")
        self.batch_nr = 0
        self.batch_size = 32
        self.client = None
        self.database = None
        self._init_db()
        self.collection: pymongo.collection.Collection = self.database[collection_name]
        self.database[collection_name+"_results"].drop()
        self.results = self.database[collection_name+"_results"]

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

    def exit(self):
        logger.info("Closing connection to database")
        self.client.close()

    def _batch_find(self, query = {}):
        data = self.collection.find(query).skip(self.batch_nr*self.batch_size).limit(self.batch_size)
        self.batch_nr += 1
        return data
    
    def batch(self, query={}):
        batch_data = self._batch_find(query)
        return list(batch_data)
    
    def transmit(self, data: list[int, str]):
        upload_dict = [{"predicted": item[0], "id": item[1]} for
                       item in data]
        self.results.insert_many(upload_dict)

    def reset(self):
        logger.info("Reseting to start")
        self.batch_nr = 0

    def get_number_of_batches(self, batch_size = None, query={}):
        if batch_size is None:
            batch_size = self.batch_size

        return round(self.collection.count_documents(query) / batch_size)

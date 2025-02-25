import os 
import pymongo
import logging
logger = logging.getLogger(__name__)


class loader():
    def __init__(self):
        logger.info("Creating Loader")
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

    def find(self, query = {}):
        return self.collection.find(query)
    
    def upload(self, item):
        self.collection.insert_many(item)

    def upload_items(self, items):
        self.collection.insert_many(items)
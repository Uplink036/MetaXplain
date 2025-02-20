import pytest 
from data import batchLoader

@pytest.fixture()
def loader():
    object = batchLoader()
    yield object
    object.exit()


class TestBatchLoader():
    def test_loader_exist(self):
        assert batchLoader() is not None
        loader = batchLoader()
        assert loader.batch_size is 32
        assert loader.batch_nr is 0 

    def test_get_batches(self, loader):
        batches = loader.get_number_of_batches()
        assert type(batches) == int
        assert (batches*loader.batch_size) == 70_000
        
    def test_get_one_data(self, loader):
        loader.batch_size = 1
        labels, data = loader.batch()
        assert len(data) == 1
        assert len(data[0]) == (28*28)
        assert len(labels) == 1
        assert labels[0] >= 0 and labels[0] < 10

    def test_get_batch_data(self, loader):
        labels, data = loader.batch()
        assert len(data) == 32
        for iteration in range(0, loader.batch_size):
            assert len(data[iteration]) == (28*28)
        assert len(labels) == 32
        for iteration in range(0, loader.batch_size):
           assert labels[0] >= 0 and labels[0] < 10

    def test_batch_find(self, loader):
        id_list = []
        for batch_nr in range(0, 5): 
            batch_data = loader._batch_find()
            for data in batch_data:
                id_list.append(data["_id"])
        set_ids = set(id_list)
        number_of_ids = len(set_ids)
        assert number_of_ids == 5*loader.batch_size 
        assert loader.batch_nr == 5

    def test_train_test_data(self, loader):
        train_batches = loader.get_number_of_batches(query={"status":"train"})
        test_batches = loader.get_number_of_batches(query={"status":"train"})
        assert train_batches > 0
        assert test_batches > 0
        assert train_batches > test_batches

    def test_train_test_batch(self, loader):
        train_labels, train_data = loader.batch({"status":"train"})
        test_labels, test_data = loader.batch({"status":"test"})
        assert len(train_data) == 32 and len(test_data) == 32
        for i in range(0, loader.batch_size):
            assert len(train_data[i]) == (28*28)
            assert len(test_data[i]) == (28*28)
        assert len(train_labels) == 32 and len(test_labels) == 32 
        for i in range(0, loader.batch_size):
           assert train_labels[i] >= 0 and train_labels[i] < 10
           assert test_labels[i] >= 0 and test_labels[i] < 10

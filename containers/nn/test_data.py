import pytest 
from data import batchLoader


@pytest.fixture()
def loader() -> batchLoader:
    return batchLoader()

class TestBatchLoader():
    def test_loader_exist(self):
        assert batchLoader() is not None
        loader = batchLoader()
        assert loader.batch_size is 32
        assert loader.batch_nr is 0 

    def test_get_batches(self, loader):
        batches = loader.get_number_of_batches()
        assert (batches*loader.batch_size) == 70_000
        
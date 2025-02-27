
import numpy as np
import pytest 
from shifting import shift

@pytest.fixture
def image():
    return np.arange(9).reshape((3,3))

def test_image_shift(image):
    shifted_image = shift(image) # Need an angle
    assert shifted_image.shape == (3,3)
    assert (image == shifted_image).all() < 9

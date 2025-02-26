
import numpy as np
import pytest 

@pytest.fixture
def image():
    return np.arange(10).reshape((3,3))

def test_image_shift(image):
    shifted_image = shift_image(image) # Need an angle
    assert shifted_image.shape == (3,3)
    assert (image == shifted_image).all() < 9


import numpy as np
import pytest 

@pytest.fixture
def image():
    return np.arange(10).reshape((3,3))

def test_image_cropping(image):
    cropped_image = noisey(image) # Add a direction
    assert cropped_image.shape == (3,3)
    assert (image == cropped_image).all() < 9

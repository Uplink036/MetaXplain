import numpy as np
import pytest 
from cropping import crop

@pytest.fixture
def image():
    return np.arange(9).reshape((3,3))

def test_image_cropping(image):
    cropped_image = crop(image) # Add a direction
    assert cropped_image.shape == (3,3)
    assert (image == cropped_image).all() < 9
    assert (cropped_image > 0).all()

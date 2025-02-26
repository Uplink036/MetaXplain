
import numpy as np
import pytest 

@pytest.fixture
def image():
    return np.arange(10).reshape((3,3))

def test_image_scaling_up(image):
    up_and_rescaled_image = upscale(image)
    assert up_and_rescaled_image.shape == (3,3)
    assert (image == up_and_rescaled_image).all() < 9

def test_image_scaling_down(image):
    down_and_rescaled_image = downscale(image)
    assert down_and_rescaled_image.shape == (3,3)
    assert (image == down_and_rescaled_image).all() < 9

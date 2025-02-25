import PIL
import numpy as np

from rotation import pil_to_np, np_to_pil, rotate_PIL, rotate

def test_rotate_pil():
    flat_matrix = [0]*9
    flat_matrix[1] = 1
    matrix = np.reshape(flat_matrix, (3, 3))
    matrix_pil = np_to_pil(matrix)
    rotated_pil = rotate_PIL(matrix_pil, 90)
    rotated_matrix = pil_to_np(rotated_pil)
    grund_truth = [0]*9
    grund_truth[3] = 1
    grund_truth = np.reshape(grund_truth, (3, 3))
    assert (grund_truth==rotated_matrix).all()


def test_rotate():
    flat_matrix = [0]*9
    flat_matrix[1] = 1
    matrix = np.reshape(flat_matrix, (3, 3))
    rotated_matrix = rotate(matrix, -90)
    grund_truth = [0]*9
    grund_truth[5] = 1
    grund_truth = np.reshape(grund_truth, (3, 3))
    assert (grund_truth==rotated_matrix).all()

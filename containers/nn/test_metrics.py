import pytest
from metrics import accuracy, recall, precision

def test_accuracy():
    targets = [1, 2, 3, 4, 5]
    predictions = [1, 2, 3, 4, 5]
    assert accuracy(targets, predictions) == 1.0
    targets = [1, 2, 3, 4, 5]
    predictions = [5, 4, 3, 2, 1]
    assert accuracy(targets, predictions) == 0.0
    targets = [1, 2, 3, 4, 5]
    predictions = [1, 2, 3, 0, 5]
    assert accuracy(targets, predictions) == 0.8
    targets = []
    predictions = []
    assert accuracy(targets, predictions) == 0.0

def test_recall():
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 0, 1, 1, 2, 2]
    assert recall(targets, predictions) == [1.0, 1.0, 1.0] + [0.0] * 7
    targets = [0, 1, 2]
    predictions = [9, 8, 7]
    assert recall(targets, predictions) == [0.0] * 10
    targets = [0, 0, 1, 1, 2, 2, 2]
    predictions = [0, 0, 1, 2, 2, 2, 2]
    assert recall(targets, predictions) == [1.0, 1.0, 1.0] + [0.0] * 7

def test_precision():
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 0, 1, 1, 2, 2]
    assert precision(targets, predictions) == [1.0, 1.0, 1.0] + [0.0] * 7
    targets = [0, 1, 2]
    predictions = [9, 8, 7]
    assert precision(targets, predictions) == [0.0] * 10
    targets = [0, 0, 1, 1, 2, 2, 2]
    predictions = [0, 1, 1, 2, 2, 2, 2]
    assert precision(targets, predictions) == [1.0, 0.5, 1.0] + [0.0] * 7
    targets = []
    predictions = []
    assert precision(targets, predictions) == [0.0] * 10
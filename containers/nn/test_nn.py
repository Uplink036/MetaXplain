import pytest 
import torch
from torch import nn
from neural_network import NeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

@pytest.fixture()
def model():
    return NeuralNetwork().to(device)

class TestNeuralNetwork():
    def test_input(self, model):
        x = torch.rand(1, 28, 28, device=device)
        logits = model(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        assert (y_pred >= 0 ) and (y_pred < 10)
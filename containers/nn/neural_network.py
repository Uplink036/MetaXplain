import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self):
        logger.info("Creating NeuralNetwork")
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 800),
            nn.ReLU(),
            nn.Linear(800, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        pred_prob = self.softmax(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred
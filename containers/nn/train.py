import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import batchLoader
from neural_network import NeuralNetwork
import torch.optim as optim


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    net = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loader = batchLoader()
    batch_size = loader.batch_size

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for iteration in range(0, loader.get_number_of_batches()):
            # get the inputs; data is a list of [inputs, labels]
            labels, inputs = loader.batch()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iteration % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
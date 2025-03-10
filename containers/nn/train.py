
import torch
import logging
from logging.handlers import RotatingFileHandler
import torch.optim as optim
from torch import nn
from data import batchLoader
from neural_network import NeuralNetwork
from metrics import accuracy, precision, recall
from tqdm import tqdm

def configure_loggers():
    logging.getLogger('').setLevel(logging.INFO)
    rotatingHandler = RotatingFileHandler(filename='/logs/rotating.log', maxBytes=4096*4, backupCount=5)
    rotatingHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotatingHandler.setFormatter(formatter)
    logging.getLogger('').addHandler(rotatingHandler)


if __name__ == "__main__":
    configure_loggers()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {device} device")
    net = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loader = batchLoader()
    batch_size = loader.batch_size

    for epoch in tqdm(range(10)):  # loop over the dataset multiple times
        running_loss = 0.0
        for iteration in tqdm(range(0, loader.get_number_of_batches(query={"status":"train"})-1)):
            # get the inputs; data is a list of [inputs, labels]
            labels, inputs = loader.batch({"status":"train"})
            inputs = torch.tensor(inputs, dtype=torch.float, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iteration % 200 == 199:
                logger.info(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        logger.info(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 200:.3f}')
        loader.reset()

    logger.info('Finished Training')

    targets = []
    predictions = []
    for iteration in tqdm(range(0, loader.get_number_of_batches(query={"status":"test"})-1)):
        labels, inputs = loader.batch({"status":"train"})
        inputs = torch.tensor(inputs, dtype=torch.float, device=device)
        outputs = net.predict(inputs)

        targets += labels
        predictions += [output.item() for output in outputs]

    logger.info("Finished Testing")
    acc = accuracy(targets, predictions)
    logger.info(f"{acc=}")
    rec = recall(targets, predictions)
    logger.info(f"{rec=}")
    pre = precision(targets, predictions)
    logger.info(f"{pre=}")


import torch
import logging
from logging.handlers import RotatingFileHandler
import torch.optim as optim
from torch import nn
from data import batchLoader
from neural_network import NeuralNetwork
from metrics import accuracy, precision, recall
from tqdm import tqdm

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def configure_loggers():
    logging.getLogger('').setLevel(logging.INFO)
    rotatingHandler = RotatingFileHandler(filename='/logs/rotating.log', maxBytes=4096*4, backupCount=5)
    rotatingHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotatingHandler.setFormatter(formatter)
    logging.getLogger('').addHandler(rotatingHandler)

def test(net: NeuralNetwork, loader: batchLoader, query={}):
    targets = []
    predictions = []
    for iteration in tqdm(range(0, loader.get_number_of_batches(query=query))):
        data = loader.batch(query)
        inputs = [item["image"] for item in data]
        labels = [item["label"] for item in data]
        ids = [item["_id"] for item in data]
        inputs = torch.tensor(inputs, dtype=torch.float, device=DEVICE)
        outputs = net.predict(inputs)
        outputs = [output.item() for output in outputs]
        targets += labels
        predictions += outputs
        loader.transmit(list(zip(outputs, ids)))
    return targets, predictions

def test_results(logger, targets, predictions):
    logger.info("Finished Testing")
    acc = accuracy(targets, predictions)
    logger.info(f"{acc=}")
    rec = recall(targets, predictions)
    logger.info(f"{rec=}")
    pre = precision(targets, predictions)
    logger.info(f"{pre=}")

if __name__ == "__main__":
    configure_loggers()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    logger.info(f"Using {DEVICE} device")
    net = NeuralNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loader = batchLoader("dataset")
    batch_size = loader.batch_size

    for epoch in tqdm(range(10)):  # loop over the dataset multiple times
        running_loss = 0.0
        for iteration in tqdm(range(0, loader.get_number_of_batches(query={"status":"train"})-1)):
            # get the inputs; data is a list of [inputs, labels]
            data = loader.batch({"status":"train"})
            inputs = [item["image"] for item in data]
            labels = [item["label"] for item in data]
            inputs = torch.tensor(inputs, dtype=torch.float, device=DEVICE)
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()

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
    loader.clean_results()

    logger.info("Starting traditonal testing")
    targets, predictions = test(net, loader, {"status":"test"})
    test_results(logger, targets, predictions)

    logger.info("Starting metamorphical testing")
    loader = batchLoader("metamorphical")
    loader.clean_results()

    targets, predictions = test(net, loader)
    test_results(logger, targets, predictions)
    logger.info("Exiting...")
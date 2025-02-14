#!/bin/bash

# Get data
curl -L -o  /tmp/data/MNIST-dataset.zip \
            https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

# Unzip data
dataset=$(find /tmp/data/*.zip)
unzip $dataset -d /tmp/data/

# Send all data to database
python3 load_data.py

# Clean up
rm -rf *idx1-ubyte
rm -rf *idx3-ubyte

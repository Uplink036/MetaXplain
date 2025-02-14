#!/bin/bash

# Get all requriments
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirments.txt
python3 train.py

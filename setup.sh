#!/bin/bash

sudo apt update

sudo apt install -y  tesseract-ocr  poppler-utils

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip 
pip install -r requirements.txt
#!/bin/bash
set -eux
pip install -e submodules/simple-knn
unzip B075X65R3X.zip
python train.py
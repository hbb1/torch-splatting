#!/bin/bash
set -eux
mkdir -p result/test
pip install -e submodules/simple-knn
unzip B075X65R3X.zip
python train.py

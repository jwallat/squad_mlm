#!bin/bash

test -d ../data || mkdir ../squad
wget -P ../data -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P ../data -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
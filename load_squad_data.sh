#!bin/bash

test -d /home/wallat/squad_mlm/data || mkdir /home/wallat/squad_mlm/data
wget -P /home/wallat/squad_mlm/data -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P /home/wallat/squad_mlm/data -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
#!/bin/sh

model_name=$1

echo $model_name.jsonnet

rm -rf models/$model_name
allennlp train configs/$model_name.jsonnet  -s models/$model_name --include-package data_readers --include-package clustering_tool
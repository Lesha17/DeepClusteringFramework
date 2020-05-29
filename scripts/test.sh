#!/usr/bin/env bash

datasets=(SearchSnippets Biomedical StackOverflow)
embedtypes=(bert_cls bert_avg bert_sif bert_max)

dataset_name=${datasets[$1 / ${#embedtypes[*]}]}
embed_name=${embedtypes[$1 % ${#embedtypes[*]}]}

echo $dataset_name $embed_name
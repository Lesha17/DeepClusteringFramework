#!/usr/bin/env bash

datasets=(SearchSnippets Biomedical StackOverflow)
embedtypes=(bert_cls bert_avg bert_sif bert_max)

echo "Slurm task id is " $SLURM_ARRAY_TASK_ID

dataset_name=${datasets[$SLURM_ARRAY_TASK_ID/${#embedtypes[*]}]}
embed_name=${embedtypes[$SLURM_ARRAY_TASK_ID%${#embedtypes[*]}]}

echo $dataset_name $embed_name

python run.py --action embed --datasets $dataset_name --embed_types $embed_name
python run.py --action train_ae --datasets $dataset_name --embed_types $embed_name
python run.py --action train_model --num_epochs 40 --start_lr 1e-3 --end_lr 1e-4 --datasets $dataset_name --embed_types $embed_name
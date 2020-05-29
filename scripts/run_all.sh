#!/usr/bin/env bash

python run.py --action embed
python run.py --action train_ae
python run.py --action train_model --num_epochs 40 --start_lr 1e-3 --end_lr 1e-4
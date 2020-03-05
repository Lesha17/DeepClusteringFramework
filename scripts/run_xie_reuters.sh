#!/usr/bin/sh

allennlp train configs/reuters_xie.jsonnet -s models/reuters_xie --include-package clustering_tool --include-package data_readers --recover
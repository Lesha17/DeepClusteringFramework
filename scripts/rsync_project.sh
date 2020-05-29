#!/bin/sh

rsync -a -e "ssh -p 2222 -i ~/.ssh/hpc_cluster" . amachnev@cluster.hpc.hse.ru:~/DeepClusteringTool --exclude='.git' --exclude=models/ \
--exclude=data/ --exclude=results/ --exclude=output/

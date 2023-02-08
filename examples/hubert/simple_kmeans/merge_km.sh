#!/bin/bash 

nshard=10
split=train-clean-100
lab_dir=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/lab/train-clean-100

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km

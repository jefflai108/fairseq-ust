#!/bin/bash 

nshard=1
split=en_train
lab_dir=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km

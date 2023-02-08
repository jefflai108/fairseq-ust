#!/bin/bash 

split=train-clean-100
nshard=10 
rank=$1
lab_dir=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/lab/train-clean-100
feat_dir=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/features/train-clean-100
km_path=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/pretrained_models/hubert_base_ls960_L9_km500.bin

python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}


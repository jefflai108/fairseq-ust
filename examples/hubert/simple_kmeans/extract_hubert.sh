#!/bin/bash 

tsv_dir=/data/sls/temp/clai24/lottery-ticket/fairseq/examples/wav2vec/data/train-clean-100
split=train-clean-100
ckpt_path=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/pretrained_models/hubert_base_ls960.pt
layer=9
nshard=10 
rank=$1
feat_dir=/data/sls/scratch/clai24/yung-sung/fairseq/examples/hubert/features/train-clean-100

python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}


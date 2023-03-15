#!/bin/bash 

tsv_dir=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es
split=en_train
ckpt_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt
layer=11
nshard=1
rank=0
feat_dir=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es

python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}



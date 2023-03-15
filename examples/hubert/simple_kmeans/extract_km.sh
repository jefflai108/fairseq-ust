#!/bin/bash 

nshard=1
rank=0
lab_dir=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es
feat_dir=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es
split=en_train
km_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}




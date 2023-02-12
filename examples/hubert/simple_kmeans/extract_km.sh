#!/bin/bash 

nshard=1
rank=0
lab_dir=/data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/simple_kmeans
feat_dir=/data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/simple_kmeans
split=es_dummy
km_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/En+Es+Fr/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}




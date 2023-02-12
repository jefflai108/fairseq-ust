#!/bin/bash 

tsv_dir=/data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/simple_kmeans
split=es_dummy
ckpt_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/En+Es+Fr/mhubert_base_vp_en_es_fr_it3.pt
layer=11
nshard=1
rank=0
feat_dir=/data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/simple_kmeans

python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}



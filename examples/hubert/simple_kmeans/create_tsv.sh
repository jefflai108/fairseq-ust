#!/bin/bash

SRC_LAN=en 
TGT_LAN=es

TGT_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_${SRC_LAN}-${TGT_LAN}
mkdir -p $TGT_DIR

python create_tsv.py \
    --audio_src_pth /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/audios \
    --tsv_pth /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC_LAN}-${TGT_LAN}/train_mined.tsv \
    --write_dir ${TGT_DIR} --lang ${SRC_LAN}


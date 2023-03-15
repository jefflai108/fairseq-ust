#!/bin/bash 

RANK=$1
NSHARD=9
stage=0

SRC_LAN=es
TGT_LAN=en

LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_${SRC_LAN}-${TGT_LAN}
mkdir -p $LAB_DIR
mkdir -p $LAB_DIR/logs

if [ $stage -eq 0 ]; then 
    python create_tsv.py \
        --audio_src_pth /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/audios \
        --tsv_pth /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC_LAN}-${TGT_LAN}/train_mined.tsv \
        --write_dir ${LAB_DIR} --lang ${SRC_LAN}
fi 
exit 0

if [ $stage -eq 1 ]; then 
    # run it on multiple GPUs
    ckpt_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt
    layer=11
    for SPLIT in ${SRC_LAN}_test ${SRC_LAN}_val ${SRC_LAN}_train ; do
        python dump_hubert_feature.py ${LAB_DIR} ${SPLIT} ${ckpt_path} ${layer} ${NSHARD} ${RANK} ${LAB_DIR}/logs &
        wait 
    done 
fi  


if [ $stage -eq 2 ]; then 
    # run it on multiple GPUs
    km_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
    for SPLIT in ${SRC_LAN}_test ${SRC_LAN}_val ${SRC_LAN}_train ; do
        python dump_km_label.py ${LAB_DIR}/logs ${SPLIT} ${km_path} ${NSHARD} ${RANK} ${LAB_DIR}/logs & 
        wait 
    done 
fi 


if [ $stage -eq 3 ]; then 
    for SPLIT in ${SRC_LAN}_test ${SRC_LAN}_val ${SRC_LAN}_train ; do
        for RANK in $(seq 0 $((NSHARD - 1))); do
            cat ${LAB_DIR}/logs/${SPLIT}_${RANK}_${NSHARD}.km
        done > ${LAB_DIR}/${SPLIT}.km
    done 
    cp /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC_LAN}-${TGT_LAN}/source_unit/dict.txt ${LAB_DIR}/dict.km.txt
fi 

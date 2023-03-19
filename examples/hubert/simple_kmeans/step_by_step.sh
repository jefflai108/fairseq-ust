#!/bin/bash 

RANK=$3
NSHARD=9
stage=-1

SRC_LAN=$1
TGT_LAN=$2

#### SRC_LAN dependent ####
# SRC_LAN=es,en,fr
ckpt_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt
layer=11
km_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
TRAIN_MIND_TSV=train_mined_t1.09

# SRC_LAN=ro 
ckpt_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_roman_it3.pt
layer=11
km_path=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_ro_it3_L11_km1000.bin
TRAIN_MIND_TSV=train_mined_t1.07
###########################

LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_${SRC_LAN}-${TGT_LAN}
mkdir -p $LAB_DIR
mkdir -p $LAB_DIR/logs

if [ $stage -eq 0 ]; then 
    # create pre-training tsv file
    python create_tsv.py \
        --audio_src_pth /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/audios \
        --s2u_dir /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC_LAN}-${TGT_LAN} \
        --train_mind_tsv ${TRAIN_MIND_TSV}.tsv --valid_vp_tsv valid_vp.tsv \
        --test_fleurs_tsv test_fleurs.tsv --test_epst_tsv test_epst.tsv \
        --write_dir ${LAB_DIR} --lang ${SRC_LAN}
fi 


if [ $stage -eq 1 ]; then 
    # run it on multiple GPUs

    # hubert feature extraction
    for SPLIT in ${SRC_LAN}-valid_vp ${SRC_LAN}-test_fleurs ${SRC_LAN}-test_epst ${SRC_LAN}-${TRAIN_MIND_TSV} ; do
        python dump_hubert_feature.py ${LAB_DIR} ${SPLIT} ${ckpt_path} ${layer} ${NSHARD} ${RANK} ${LAB_DIR}/logs &
        wait 
    done 
    
    # hubert km unit extraction 
    for SPLIT in ${SRC_LAN}-valid_vp ${SRC_LAN}-test_fleurs ${SRC_LAN}-test_epst ${SRC_LAN}-${TRAIN_MIND_TSV} ; do
        python dump_km_label.py ${LAB_DIR}/logs ${SPLIT} ${km_path} ${NSHARD} ${RANK} ${LAB_DIR}/logs & 
        wait 
    done 
fi 


if [ $stage -eq 3 ]; then 
    # merge km files 
    for SPLIT in ${SRC_LAN}-valid_vp ${SRC_LAN}-test_fleurs ${SRC_LAN}-test_epst ${SRC_LAN}-${TRAIN_MIND_TSV} ; do
        for RANK in $(seq 0 $((NSHARD - 1))); do
            cat ${LAB_DIR}/logs/${SPLIT}_${RANK}_${NSHARD}.km
        done > ${LAB_DIR}/${SPLIT}.km
    done 
    cp /data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC_LAN}-${TGT_LAN}/source_unit/dict.txt ${LAB_DIR}/dict.km.txt
fi 

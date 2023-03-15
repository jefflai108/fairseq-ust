#!/bin/bash 

PROC_FLEURS_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/fairseq_processed_fleurs
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
HUBERT_CKPT=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/dummy/checkpoints/checkpoint_best.pt

export PYTHONPATH="${PYTHONPATH}:${PWD}/../../:${PWD}/simple_kmeans"

#CUDA_VISIBLE_DEVICES=0 
python3 hubert_info_align_decode.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-ckpt ${HUBERT_CKPT}


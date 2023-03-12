#!/bin/bash 

PROC_FLEURS_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/fairseq_processed_fleurs
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
HUBERT_MODEL_DIR=/data/sls/temp/clai24/pretrained-models/mHuBERT

export PYTHONPATH="${PYTHONPATH}:${PWD}/../../:${PWD}/simple_kmeans"

#CUDA_VISIBLE_DEVICES=0 
python3 hubert_inference.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-model-dir ${HUBERT_MODEL_DIR}


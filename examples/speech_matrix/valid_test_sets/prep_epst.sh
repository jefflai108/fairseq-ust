#!/bin/bash 

EPST_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/europarl_st/v1.1
PROC_EPST_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/europarl_st/fairseq_processed
SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit

# EPST_DIR: the directory of original EPST data.
# PROC_EPST_DIR: the directory of EPST processed data.
# SAVE_ROOT: the directory to save SpeechMatrix mined data
python3 prep_epst_test_data.py \
    --epst-dir ${EPST_DIR} \
    --proc-epst-dir ${PROC_EPST_DIR} \
    --save-root ${SAVE_ROOT}

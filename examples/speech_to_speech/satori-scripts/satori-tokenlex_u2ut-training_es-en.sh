#!/bin/bash

SRC=${1:-es}
TGT=${2:-en}
MULTI_GPU=${3:-true}
L=${4:-50}

if [ "$MULTI_GPU" = true ]; then
    UPDATE_FREQ=1
else
    UPDATE_FREQ=4
fi

DATA_ROOT=/nobackup/users/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}
LEXICON_ROOT=/nobackup/users/clai24/data/speech_matrix/speech_to_unit/lexicon_alignment/${SRC}-${TGT}

if [ $L -eq 50 ]; then
    ######## L <= 50 #######
    TRAIN_SET="train_mined_t1.09_filter50_u2u"
    VALID_SET="valid_vp_filter50_u2u"
    LEX_ALIGN_FILE="diag.align.filter50_u2u_probt0.1.npy"
fi

if [ $L -eq 100 ]; then
    ######## L <= 100 #######
    TRAIN_SET="train_mined_t1.09_filter100_u2u"
    VALID_SET="valid_vp_filter100_u2u"
    LEX_ALIGN_FILE="diag.align.filter100_u2u_probt0.1.npy"
fi

if [ $L -eq 200 ]; then
    ######## L <= 200 #######
    TRAIN_SET="train_mined_t1.09_filter200_u2u"
    VALID_SET="valid_vp_filter200_u2u"
    LEX_ALIGN_FILE="diag.align.filter200_u2u_probt0.1.npy"
fi

if [ $L -eq 250 ]; then
    ######## L <= 250 #######
    TRAIN_SET="train_mined_t1.09_filter250_u2u"
    VALID_SET="valid_vp_filter250_u2u"
    LEX_ALIGN_FILE="diag.align.filter250_u2u_probt0.1.npy"
fi

if [ $L -eq 400 ]; then
    ######## L <= 400 #######
    TRAIN_SET="train_mined_t1.09_filter400_u2u"
    VALID_SET="valid_vp_filter400_u2u"
    LEX_ALIGN_FILE="diag.align.filter400_u2u_probt0.1.npy"
fi

if [ $L -eq 500 ]; then
    ######## L <= 500 #######
    TRAIN_SET="train_mined_t1.09_filter500_u2u"
    VALID_SET="valid_vp_filter500_u2u"
    LEX_ALIGN_FILE="diag.align.filter500_u2u_probt0.1.npy"
fi

if [ $L -eq 1024 ]; then
    ######## L <= 1k #######
    TRAIN_SET="train_mined_t1.09_filter1024_u2u"
    VALID_SET="valid_vp_filter800_u2u"
    LEX_ALIGN_FILE="diag.align.filter1024_u2u_probt0.1.npy"
fi

MODEL_DIR=/nobackup/users/clai24/lexicon/exp/bilingual_textless_s2st/${SRC}-${TGT}/v0-${TRAIN_SET}_diag.align.probt0.1
mkdir -p ${MODEL_DIR}

# reduce "max-update" from 400000 to speedup model development.
# based on our initial training run, 25k steps should suffice for `train_mined_t1.09_filter100`
# added "--no-epoch-checkpoints' to avoid saving intermediate ckpts
# experimenting for `train_mined_t1.09_filter{200,250,400,500,1024}` now. Guess 50k steps suffice.
# We are using a smaller speech encoder by setting "--arch s2ut_transformer_fisher". For fair comparison w.r.t SpeechMatrix, switch to "--arch s2ut_transformer"
# removed "--multitask-config-yaml config_multitask.yaml" as we use src unit has input
# reduce --max-tokens from 20k to 14k for CUDA mem error: 20k for L<=400, 16k for L<=500, 14k for L<=1024
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task token_lexical_unit_to_unit --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --source-is-code --source-code-size 1000 \
  --is-copy --lex-alignment-npy ${LEXICON_ROOT}/${LEX_ALIGN_FILE} \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch token_lex_u2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset ${TRAIN_SET} --valid-subset ${VALID_SET} \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 50000 --max-tokens 14000 --max-target-positions 3000 --update-freq ${UPDATE_FREQ} \
  --seed 1 --fp16 --num-workers 8 \
  2>&1 | tee ${MODEL_DIR}/train.log

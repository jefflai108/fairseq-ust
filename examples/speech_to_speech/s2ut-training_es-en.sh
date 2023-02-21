#!/bin/bash 

SRC=es
TGT=en
MULTI_GPU=true

if [ "$MULTI_GPU" = true ]; then 
    UPDATE_FREQ=1
else
    UPDATE_FREQ=4
fi 

DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}
TRAIN_SET="train_mined_t1.09"
MODEL_DIR=/data/sls/scratch/clai24/lexicon/exp/bilingual_textless_s2st/${SRC}-${TGT}/v0-${TRAIN_SET}

fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset ${TRAIN_SET} --valid-subset "valid_vp" \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq ${UPDATE_FREQ} \
  --seed 1 --fp16 --num-workers 8


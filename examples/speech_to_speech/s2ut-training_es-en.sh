#!/bin/bash 

SRC=${1:-es}
TGT=${2:-en}
MULTI_GPU=${3:-true}

if [ "$MULTI_GPU" = true ]; then 
    UPDATE_FREQ=1
else
    UPDATE_FREQ=4
fi 

DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}

######## L <= 100 #######
TRAIN_SET="train_mined_t1.09_filter100" 
VALID_SET="valid_vp_filter100"

######## L <= 200 #######
TRAIN_SET="train_mined_t1.09_filter200" 
VALID_SET="valid_vp_filter200"

MODEL_DIR=/data/sls/scratch/clai24/lexicon/exp/bilingual_textless_s2st/${SRC}-${TGT}/v0-${TRAIN_SET}

# reduce "max-update" from 400000 to speedup model development. 
# based on our initial training run, 25k steps should suffice for `train_mined_t1.09_filter100`
# added "--no-epoch-checkpoints' to avoid saving intermediate ckpts
# experimenting for `train_mined_t1.09_filter200` now. Guess 40k steps suffice. 
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset ${TRAIN_SET} --valid-subset ${VALID_SET} \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 40000 --max-tokens 20000 --max-target-positions 3000 --update-freq ${UPDATE_FREQ} \
  --seed 1 --fp16 --num-workers 8 

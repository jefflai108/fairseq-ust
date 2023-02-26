#!/bin/bash 

SRC=${1:-es}
TGT=${2:-en}
MULTI_GPU=${3:-true}
L=${4:-100}

if [ "$MULTI_GPU" = true ]; then
    UPDATE_FREQ=1
else
    UPDATE_FREQ=4
fi

DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}

if [ $L -eq 100 ]; then
    ######## L <= 100 #######
    TRAIN_SET="train_mined_t1.09_filter100_u2u"
    VALID_SET="valid_vp_filter100_u2u"
fi

if [ $L -eq 200 ]; then
    ######## L <= 200 #######
    TRAIN_SET="train_mined_t1.09_filter200_u2u"
    VALID_SET="valid_vp_filter200_u2u"
fi

if [ $L -eq 250 ]; then
    ######## L <= 250 #######
    TRAIN_SET="train_mined_t1.09_filter250_u2u"
    VALID_SET="valid_vp_filter250_u2u"
fi

if [ $L -eq 400 ]; then
    ######## L <= 400 #######
    TRAIN_SET="train_mined_t1.09_filter400_u2u"
    VALID_SET="valid_vp_filter400_u2u"
fi

if [ $L -eq 500 ]; then
    ######## L <= 500 #######
    TRAIN_SET="train_mined_t1.09_filter500_u2u"
    VALID_SET="valid_vp_filter500_u2u"
fi

if [ $L -eq 1024 ]; then
    ######## L <= 1k #######
    TRAIN_SET="train_mined_t1.09_filter1024_u2u"
    VALID_SET="valid_vp_filter800_u2u"
fi 

MODEL_DIR=/data/sls/scratch/clai24/lexicon/exp/bilingual_textless_s2st/${SRC}-${TGT}/v0-${TRAIN_SET}
mkdir -p ${MODEL_DIR}

# reduce "max-update" from 400000 to speedup model development. 
# based on our initial training run, 25k steps should suffice for `train_mined_t1.09_filter100`
# added "--no-epoch-checkpoints' to avoid saving intermediate ckpts
# experimenting for `train_mined_t1.09_filter{200,250,400,500,1024}` now. Guess 50k steps suffice.
# removed "--multitask-config-yaml config_multitask.yaml" as we use src unit has input 
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task lexical_speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --source-is-code --source-code-size 1000 \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch u2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset ${TRAIN_SET} --valid-subset ${VALID_SET} \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 50000 --max-tokens 20000 --max-target-positions 3000 --update-freq ${UPDATE_FREQ} \
  --seed 1 --fp16 --num-workers 8 \
  2>&1 | tee ${MODEL_DIR}/train.log

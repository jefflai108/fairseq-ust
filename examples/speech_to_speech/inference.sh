#!/bin/bash 

# SpeechMatrix es-en 
DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en
GEN_SUBSET=test_epst 
RESULTS_PATH=dummy
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --path /data/sls/temp/clai24/pretrained-models/bilingual_textless_s2st/checkpoint_textless_es_en.pt  --gen-subset $GEN_SUBSET \
  --max-tokens 50000 \
  --beam 10 --max-len-a 1 \
  --results-path ${RESULTS_PATH}

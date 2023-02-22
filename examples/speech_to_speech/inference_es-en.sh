#!/bin/bash 

stage=0

SRC=${1:-es}
TGT=${2:-en}

BEAM=10
DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}
GEN_SUBSET=test_epst 
RESULTS_PATH=/data/sls/temp/clai24/data/speech_matrix/textless_s2ut_gen/${SRC}-${TGT}_beam${BEAM}
WAVE_PATH=${RESULTS_PATH}/waveforms
mkdir -p ${WAVE_PATH}
VOCODER_CKPT=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/vocoder_${TGT}.pt
VOCODER_CFG=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/config_${TGT}.json

if [ $stage -eq 0 ]; then 
    # textless S2UT model inference
    fairseq-generate $DATA_ROOT \
      --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
      --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan \
      --path /data/sls/temp/clai24/pretrained-models/bilingual_textless_s2st/checkpoint_textless_${SRC}_${TGT}.pt --gen-subset $GEN_SUBSET \
      --max-tokens 50000 \
      --beam $BEAM --max-len-a 1 \
      --results-path ${RESULTS_PATH}
fi 

if [ $stage -le 1 ]; then 
    # unit-to-speech vocoder synthesis
    grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
      sed 's/^D-//ig' | sort -nk1 | cut -f3 \
      > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

    python generate_waveform_from_code.py \
      --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
      --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
      --results-path ${WAVE_PATH} --dur-prediction
fi 

if [ $stage -le 2 ]; then 
    # ASR-BLEU eval 
    REFERENCE_TEXT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}/${GEN_SUBSET}.${TGT}
    python asr_bleu/compute_asr_bleu.py --lang ${TGT} \
        --audio_dirpath ${WAVE_PATH} \
        --reference_path ${REFERENCE_TEXT} \
        --reference_format txt \
        --results_dirpath ${RESULTS_PATH} \
        --transcripts_path ${RESULTS_PATH}/asr_transcription_text
fi

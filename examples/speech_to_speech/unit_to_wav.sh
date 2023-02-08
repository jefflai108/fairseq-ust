#!/bin/bash 

LAN=es
LAN=en

IN_CODE_FILE=UNIT_TO_WAVEFORM_FILES/${LAN}_pred_hubert_units
VOCODER_CKPT=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/g_00500000
VOCODER_CFG=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/config.json
RESULTS_PATH=UNIT_TO_WAVEFORM_FILES/${LAN}_pred_waveform

################# for debugging token-level lexicon alignment ##################
IN_CODE_FILE=UNIT_TO_WAVEFORM_FILES/es-en_ibm2_lexicon/${LAN}_aligned_tokens
VOCODER_CKPT=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/g_00500000
VOCODER_CFG=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/config.json
RESULTS_PATH=UNIT_TO_WAVEFORM_FILES/es-en_ibm2_lexicon/${LAN}_pred_waveform
################################################################################


################# for generating speech from Ekin translation models ##################
LAN_PAIR=es-en
SYS=lstm.filter200
SYS=lexlstm.filter200
TAR_LAN=en

IN_CODE_FILE=UNIT_TO_WAVEFORM_FILES/${LAN_PAIR}_${SYS}.gold_${TAR_LAN}
IN_CODE_FILE=UNIT_TO_WAVEFORM_FILES/${LAN_PAIR}_${SYS}.pred_${TAR_LAN}
VOCODER_CKPT=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/g_00500000
VOCODER_CFG=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/${LAN}_unit_hifigan/config.json
RESULTS_PATH=UNIT_TO_WAVEFORM_FILES/waveforms/${LAN_PAIR}_${SYS}.gold_${TAR_LAN}
RESULTS_PATH=UNIT_TO_WAVEFORM_FILES/waveforms/${LAN_PAIR}_${SYS}.pred_${TAR_LAN}

mkdir -p $RESULTS_PATH
################################################################################

python generate_waveform_from_code.py \
  --in-code-file ${IN_CODE_FILE} \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} \
  --results-path ${RESULTS_PATH} --dur-prediction

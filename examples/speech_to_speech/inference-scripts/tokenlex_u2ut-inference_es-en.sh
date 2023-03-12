#!/bin/bash 

stage=0

SRC=${1:-es}
TGT=${2:-en}
L=${3:-50}
LEXICON_T=${4:-0.1}

BEAM=10
DATA_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}
LEXICON_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/lexicon_alignment/${SRC}-${TGT}
GEN_SUBSET=test_epst 

if [ $L -eq 50 ]; then
    ######## L <= 50 #######
    TRAIN_SET="train_mined_t1.09_filter50_u2u"
    VALID_SET="valid_vp_filter50_u2u"
    LEX_ALIGN_FILE="diag.align.filter50_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 100 ]; then
    ######## L <= 100 #######
    TRAIN_SET="train_mined_t1.09_filter100_u2u"
    VALID_SET="valid_vp_filter100_u2u"
    LEX_ALIGN_FILE="diag.align.filter100_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 200 ]; then
    ######## L <= 200 #######
    TRAIN_SET="train_mined_t1.09_filter200_u2u"
    VALID_SET="valid_vp_filter200_u2u"
    LEX_ALIGN_FILE="diag.align.filter200_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 250 ]; then
    ######## L <= 250 #######
    TRAIN_SET="train_mined_t1.09_filter250_u2u"
    VALID_SET="valid_vp_filter250_u2u"
    LEX_ALIGN_FILE="diag.align.filter250_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 400 ]; then
    ######## L <= 400 #######
    TRAIN_SET="train_mined_t1.09_filter400_u2u"
    VALID_SET="valid_vp_filter400_u2u"
    LEX_ALIGN_FILE="diag.align.filter400_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 500 ]; then
    ######## L <= 500 #######
    TRAIN_SET="train_mined_t1.09_filter500_u2u"
    VALID_SET="valid_vp_filter500_u2u"
    LEX_ALIGN_FILE="diag.align.filter500_u2u_probt${LEXICON_T}.npy"
fi

if [ $L -eq 1024 ]; then
    ######## L <= 1k #######
    TRAIN_SET="train_mined_t1.09_filter1024_u2u"
    VALID_SET="valid_vp_filter800_u2u"
    LEX_ALIGN_FILE="diag.align.filter1024_u2u_probt${LEXICON_T}.npy"
fi

############### our own model trained on filtered data ################
TRAINED_S2S_MODEL=/data/sls/scratch/clai24/lexicon/exp/bilingual_textless_s2st/${SRC}-${TGT}/v0-${TRAIN_SET}_diag.align.probt${LEXICON_T}/checkpoint_best.pt
RESULTS_PATH=/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/${SRC}-${TGT}_v0-train_mined_t1.09_filter${L}_u2u_diag.align.probt${LEXICON_T}_beam${BEAM}/
#######################################################################

WAVE_PATH=${RESULTS_PATH}/waveforms
mkdir -p ${WAVE_PATH}
VOCODER_CKPT=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/vocoder_${TGT}.pt
VOCODER_CFG=/data/sls/temp/clai24/data/speech_matrix/unit_vocoder/config_${TGT}.json

#for ORIG_GEN_SUBSET in valid_vp_filter${L}; do 
for ORIG_GEN_SUBSET in test_epst test_epst_filter${L} test_fleurs; do

GEN_SUBSET=${ORIG_GEN_SUBSET}_u2u

if [ $stage -eq 0 ]; then 
    # textless S2UT model inference
    fairseq-generate $DATA_ROOT \
      --config-yaml config.yaml \
      --task token_lexical_unit_to_unit --target-is-code --target-code-size 1000 --vocoder code_hifigan \
      --source-is-code --source-code-size 1000 \
      --is-copy --lex-alignment-npy ${LEXICON_ROOT}/${LEX_ALIGN_FILE} \
      --path ${TRAINED_S2S_MODEL} --gen-subset $GEN_SUBSET \
      --max-tokens 20000 \
      --beam $BEAM --max-len-a 1 \
      --results-path ${RESULTS_PATH} 
fi 

if [ $stage -le 1 ]; then 
    # ensure no pre-existing waves
    [ -d ${WAVE_PATH} ] && rm -r ${WAVE_PATH}

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
    REFERENCE_TEXT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/${SRC}-${TGT}/${ORIG_GEN_SUBSET}.${TGT}
    python asr_bleu/compute_asr_bleu.py --lang ${TGT} \
        --audio_dirpath ${WAVE_PATH} \
        --reference_path ${REFERENCE_TEXT} \
        --reference_format txt \
        --reference_split ${GEN_SUBSET} \
        --results_dirpath ${RESULTS_PATH} \
        --transcripts_path ${RESULTS_PATH}/asr_transcription_${GEN_SUBSET}
fi

done 

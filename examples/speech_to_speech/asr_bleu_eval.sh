#!/bin/bash 

LANG=en
PATH_TO_AUDIO_DIR=UNIT_TO_WAVEFORM_FILES/es-en_lexlstm.filter200.pred_en


python asr_bleu/compute_asr_bleu.py --lang en \
--audio_dirpath <PATH_TO_AUDIO_DIR> \
--reference_path <PATH_TO_REFERENCES_FILE> \
--reference_format txt

#!/bin/bash 

LANG=en
PATH_TO_AUDIO_DIR=/data/sls/temp/clai24/data/speech_matrix/textless_s2ut_gen/es-en_beam10/waveforms
PATH_TO_REFERENCES_FILE=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/test_epst.en

python asr_bleu/compute_asr_bleu.py --lang en \
    --audio_dirpath $PATH_TO_AUDIO_DIR \
    --reference_path $PATH_TO_REFERENCES_FILE \
    --reference_format txt \
    --results_dirpath /data/sls/temp/clai24/data/speech_matrix/textless_s2ut_gen/es-en_beam10 \
    --transcripts_path /data/sls/temp/clai24/data/speech_matrix/textless_s2ut_gen/es-en_beam10/prediction_text

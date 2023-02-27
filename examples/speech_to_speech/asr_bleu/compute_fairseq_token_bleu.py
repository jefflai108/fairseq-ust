from sacrebleu.metrics import BLEU, CHRF, TER
import pandas as pd

def preprocess_training_log(fairseq_pred_unit, ground_truth_tsv): 

    with open(fairseq_pred_unit, 'r') as f: 
        content = f.readlines() 
    pred_units = [x.strip('\n') for x in content]

    src_df = pd.read_table(ground_truth_tsv, sep='\t', header=0)

    print(src_df['tgt_audio'])
    num_examples = len(src_df['tgt_audio'])

    refs, sys = [], []
    for i in range(num_examples):
        pred1 = pred_units[i]
        sys.append(pred1)

        gold1 = src_df['tgt_audio'][i]
        refs.append(gold1) 

    refs = [refs]
    assert len(refs) == 1
    print(f'There are {len(sys)} utterances')

    return refs, sys 


def compute_bleu(refs, sys): 
    bleu = BLEU()
    score = bleu.corpus_score(sys, refs)
    print(score)

if __name__ == '__main__': 

    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_FAIR_beam10/generate-valid_vp_filter1000.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter1000.tsv') # BLEU = 25.91 57.6/34.2/21.0/12.9 (BP = 0.960 ratio = 0.961 hyp_len = 336437 ref_len = 350261)

    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter100_beam10/generate-valid_vp_filter100.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter100.tsv') # BLEU = 17.46 38.1/22.5/14.1/8.9 (BP = 0.965 ratio = 0.966 hyp_len = 4169 ref_len = 4317)

    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter200_beam10/generate-valid_vp_filter200.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter200.tsv') # BLEU = 23.33 52.0/31.6/20.0/12.9 (BP = 0.913 ratio = 0.917 hyp_len = 27723 ref_len = 30243)

    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter250_beam10/generate-valid_vp_filter250.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter250.tsv') # BLEU = 22.96 51.3/30.7/19.1/12.1 (BP = 0.934 ratio = 0.936 hyp_len = 53886 ref_len = 57576)
 
    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter400_beam10/generate-valid_vp_filter400.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter400.tsv') # BLEU = 24.33 54.2/32.5/20.3/12.7 (BP = 0.937 ratio = 0.939 hyp_len = 155683 ref_len = 165825)

    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter500_beam10/generate-valid_vp_filter500.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter500.tsv') # BLEU = 24.57 54.1/32.3/20.0/12.5 (BP = 0.956 ratio = 0.957 hyp_len = 206639 ref_len = 215853)
    
    #refs, sys = preprocess_training_log('/data/sls/scratch/clai24/lexicon/exp/textless_s2ut_gen/es-en_v0-train_mined_t1.09_filter1024_beam10/generate-valid_vp_filter1024.unit', 
    #                                    '/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en/valid_vp_filter1024.tsv') # BLEU = 23.90 55.8/33.3/20.5/12.7 (BP = 0.905 ratio = 0.909 hyp_len = 320367 ref_len = 352289)

    compute_bleu(refs, sys)

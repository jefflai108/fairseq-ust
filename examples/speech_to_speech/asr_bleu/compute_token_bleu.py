from sacrebleu.metrics import BLEU, CHRF, TER

def preprocess_training_log(log_file): 

    with open(log_file, 'r') as f: 
        content = f.readlines() 
    content = [x.strip('\n') for x in content]

    refs, sys = [], []
    for x in content: 
        if len(x.split()) >= 4:
            if 'gold' in x.split()[3]: 
                gold1 = x.split('gold ')[1]
                gold1 = gold1.strip('][').replace('\\n', '').split("', '")
                gold1 = ' '.join(gold1).replace("'", '')
                print(gold1)
                refs.append(gold1)
            elif 'pred' in x.split()[3]:
                pred1 = x.split('pred ')[1]
                pred1 = pred1.strip('][').replace('\\n', '').split("', '")
                pred1 = ' '.join(pred1).replace("'", '')
                print(pred1)
                sys.append(pred1)
  
    refs = [refs]
    assert len(refs) == 1
    print(f'There are {len(sys)} utterances')

    return refs, sys 


def compute_bleu(refs, sys): 
    bleu = BLEU()
    score = bleu.corpus_score(sys, refs)
    print(score)

if __name__ == '__main__': 
    #refs, sys = preprocess_training_log('/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/ekin/from_ekin/es-en/lstm.filter120.pred') # BLEU = 17.50 37.7/22.0/13.5/8.4 (BP = 1.000 ratio = 1.045 hyp_len = 7836 ref_len = 7499)
    #refs, sys = preprocess_training_log('/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/ekin/from_ekin/es-en/lexlstm.filter120.pred') # BLEU = 17.07 37.3/21.3/13.0/8.2 (BP = 1.000 ratio = 1.012 hyp_len = 7591 ref_len = 7499)
    #refs, sys = preprocess_training_log('/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/ekin/from_ekin/es-en/lstm.filter200.pred') # BLEU = 18.33 40.5/22.9/13.9/8.7 (BP = 1.000 ratio = 1.070 hyp_len = 32365 ref_len = 30243)
    #refs, sys = preprocess_training_log('/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/ekin/from_ekin/es-en/lexlstm.filter200.pred') # BLEU = 17.70 39.4/22.3/13.5/8.3 (BP = 1.000 ratio = 1.080 hyp_len = 32662 ref_len = 30243)

    compute_bleu(refs, sys)

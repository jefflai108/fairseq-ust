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
  
    print(f'There are {len(sys)} utterances')

    return refs, sys 

def write_to_file(refs, sys, pred_f, gold_f): 

    with open(gold_f, 'w') as f: 
        for ref in refs: 
            f.write('%s\n' % ref) 

    with open(pred_f, 'w') as f: 
        for sy in sys: 
            f.write('%s\n' % sy) 

if __name__ == '__main__': 

    lan_pair = 'es-en'

    system   = 'lstm.filter120'
    system   = 'lexlstm.filter120'
    system   = 'lstm.filter200'
    system   = 'lexlstm.filter200'

    refs, sys = preprocess_training_log(f'/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/ekin/from_ekin/{lan_pair}/{system}.pred') # BLEU = 17.50 37.7/22.0/13.5/8.4 (BP = 1.000 ratio = 1.045 hyp_len = 7836 ref_len = 7499)

    write_to_file(refs, sys, 
                f'UNIT_TO_WAVEFORM_FILES/{lan_pair}_{system}.pred_' + lan_pair.split('-')[1], 
                f'UNIT_TO_WAVEFORM_FILES/{lan_pair}_{system}.gold_' + lan_pair.split('-')[1]
                )

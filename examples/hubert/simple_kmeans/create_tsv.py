import os, logging
import random 
import pandas as pd 

from fairseq_extract_waveform import get_features_or_waveform

def main(audio_src_pth, s2u_dir, train_mind_tsv, valid_vp_tsv, test_fleurs_tsv, test_epst_tsv, write_dir, lang):
 
    # valid and test
    for tsv_name in [valid_vp_tsv, test_fleurs_tsv, test_epst_tsv]: 
        tsv_pth = os.path.join(s2u_dir, tsv_name)
        df = pd.read_csv(tsv_pth, sep='\t', header=None, skiprows=1)
        content = df.iloc[:, 1]
        if tsv_name == valid_vp_tsv:
            iterate_and_write(audio_src_pth, content, os.path.join(write_dir, lang + '-' + tsv_name))
        else:
            iterate_and_write('/data/../', content, os.path.join(write_dir, lang + '-' + tsv_name))

    # train 
    tsv_pth = os.path.join(s2u_dir, train_mind_tsv)
    df = pd.read_csv(tsv_pth, sep='\t', header=None, skiprows=1)
    content = df.iloc[:, 2]
    iterate_and_write(audio_src_pth, content, os.path.join(write_dir, lang + '-' + train_mind_tsv))

    
    ## train : val : test = 0.8 : 0.1 : 0.1 
    #split_index1 = int(0.8 * len(content))
    #split_index2 = int(0.9 * len(content))
    #train_content = content[:split_index1]
    #val_content  = content[split_index1:split_index2]
    #test_content = content[split_index2:]

    #iterate_and_write(audio_src_pth, train_content, os.path.join(write_dir, target_lang + '_train.tsv'))
    #iterate_and_write(audio_src_pth, val_content, os.path.join(write_dir, target_lang + '_val.tsv'))
    #iterate_and_write(audio_src_pth, test_content, os.path.join(write_dir, target_lang + '_test.tsv'))

def iterate_and_write(audio_src_pth, content, write_pth):
    f = open(write_pth, 'w') 
    f.write('%s\n' % audio_src_pth)
    for utt_id in content: 
        frames = extract_single_utt_waveform(audio_src_pth, utt_id)
        f.write('%s\t%d\n' % (utt_id, frames))
    f.close()

def extract_single_utt_waveform(audio_src_pth, utt_identifier):
    feat = get_features_or_waveform(os.path.join(audio_src_pth, utt_identifier), need_waveform=True, use_sample_rate=16000)
    return feat.shape[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_src_pth", type=str, 
                        default="/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/audios")
    parser.add_argument("--s2u_dir", type=str, 
                        default="/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/en-es")
    parser.add_argument("--train_mind_tsv", type=str, 
                        default="train_mined_t1.09.tsv")
    parser.add_argument("--valid_vp_tsv", type=str, 
                        default="valid_vp.tsv")
    parser.add_argument("--test_fleurs_tsv", type=str, 
                        default="test_fleurs.tsv")
    parser.add_argument("--test_epst_tsv", type=str, 
                        default="test_epst.tsv")
    parser.add_argument("--write_dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es")
    parser.add_argument("--lang", type=str)

    args = parser.parse_args()
    logging.info(str(args))
    main(**vars(args))

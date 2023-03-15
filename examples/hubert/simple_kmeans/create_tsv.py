import os, logging
import random 
import pandas as pd 

from fairseq_extract_waveform import get_features_or_waveform

def main(audio_src_pth, tsv_pth, write_dir, target_lang):
    df = pd.read_csv(tsv_pth, sep='\t', header=None, skiprows=1)
    content = df.iloc[:, 2]

    # train : val : test = 0.8 : 0.1 : 0.1 
    split_index1 = int(0.8 * len(content))
    split_index2 = int(0.9 * len(content))
    train_content = content[:split_index1]
    val_content  = content[split_index1:split_index2]
    test_content = content[split_index2:]

    iterate_and_write(audio_src_pth, train_content, os.path.join(write_dir, target_lang + '_train.tsv'))
    iterate_and_write(audio_src_pth, val_content, os.path.join(write_dir, target_lang + '_val.tsv'))
    iterate_and_write(audio_src_pth, test_content, os.path.join(write_dir, target_lang + '_test.tsv'))

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
    parser.add_argument("--tsv_pth", type=str, 
                        default="/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/en-es/train_mined.tsv")
    parser.add_argument("--write_dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es")
    parser.add_argument("--lang", type=str, 
                        default="en")

    args = parser.parse_args()
    logging.info(str(args))

    main(args.audio_src_pth, args.tsv_pth, args.write_dir, args.lang)
    

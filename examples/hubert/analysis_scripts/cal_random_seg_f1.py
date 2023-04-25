import os
import pickle 
import copy
from collections import defaultdict

import numpy as np

from utils import *

class WordSegF1(): 
    
    def __init__(self, args):
        self.args = args 
        self.frame_rate = 50

        # setup data
        self.get_uttid_list()
        self.read_segmentation_pickle()
        self.get_pickles()

        # span pre-process + cal word_seg F1
        self.cal_corpus_f1()

    def read_segmentation_pickle(self): 
        with open(os.path.join(self.args.mfa_dir, f"{self.args.split}-phones_seg.pkl"), "rb") as file:
            self.utt2phnseg = pickle.load(file)
        with open(os.path.join(self.args.mfa_dir, f"{self.args.split}-words_seg.pkl"), "rb") as file:
            self.utt2wrdseg = pickle.load(file)

    def get_pickles(self): 
        self.utt2span = {}
        for uttid in self.uttids:
            if self.args.parse_alg == "top_down":
                span_pkl = f'{self.args.parse_out_dir}/v03.{self.args.split}.{self.args.parse_alg}.random/{uttid}-minsplitpmi{self.args.min_pmi}.pkl'
            elif self.args.parse_alg == "bottom_up":
                span_pkl = f'{self.args.parse_out_dir}/v03.{self.args.split}.{self.args.parse_alg}.random/{uttid}-minmergepmi{self.args.min_pmi}.pkl'
            self.utt2span[uttid] = self.read_parse_pickle(span_pkl, uttid)

    def read_parse_pickle(self, span_pkl, uttid): 
        with open(span_pkl, "rb") as file:
            my_dict = pickle.load(file)
            if self.args.parse_alg == "top_down":
                oracle_phone_seg = self.utt2phnseg[uttid]
                my_list = [x[0] for x in my_dict] # this is the indidces of phone segments
                my_dict = []
                for t in my_list:
                    start_seg_id, end_seg_id = t
                    start_sec = oracle_phone_seg[start_seg_id][1]
                    end_sec = oracle_phone_seg[end_seg_id][2]
                    my_dict.append((start_sec, end_sec))
        return my_dict

    def get_uttid_list(self): 
        import csv 

       # Open CSV file and read first column
        with open(os.path.join(self.args.km_dir, f"{self.args.split}.tsv"), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            filenames = [row[0].split('\t')[0].split('/')[-1] for row in reader]

            trimmed_filenames = filenames
            len_extensions = 0
            if filenames[0].endswith('.flac'): 
                len_extensions = len('.flac')
            elif filenames[0].endswith('.wav'):
                len_extensions = len('.wav')
            
            if len_extensions > 0: 
                trimmed_filenames = [filename[:-len_extensions] for filename in filenames]
            
        self.uttids = trimmed_filenames

    def cal_corpus_f1(self):
        all_f1 = 0
        all_tp, all_fp, all_fn = 0, 0, 0
        for utt in self.uttids:
            my_dict = self.utt2span[utt]
            if self.args.parse_alg == "bottom_up":
                span_levels = self.process_bottomup_spans(my_dict, utt)
            elif self.args.parse_alg == "top_down":
                span_levels = self.process_topdown_spans(my_dict, utt)
            iter_lst = span_levels.items()
            total_level = len(iter_lst)
            
            best_f1, best_split, level = 0, None, 0
            best_tp, best_fp, best_fn = 0, 0, 0
            for level, predicted_segs in iter_lst:   
                # ChatGPT's word segmentation is based on segments overlap 
                # We will adopt the stricter criterion used on Buckeye, which is based on the distance 
                # from oracle boundaries. 
                #f1_score, (tp, fp, fn) = self.cal_chatgpt_word_seg_f1(predicted_segs, utt)
                f1_score, (tp, fp, fn) = self.buckeye_word_seg_f1(predicted_segs, utt)
                if f1_score > best_f1: 
                    best_f1 = f1_score
                    best_split = level 
                    best_tp = tp
                    best_fp = fp
                    best_fn = fn

            print(f"best word seg F1 score is {best_f1:.2f} at merge level {best_split} out of {total_level}")
            all_f1 += best_f1
            all_tp += best_tp
            all_fp += best_fp 
            all_fn += best_fn 
        
        corpus_f1 = f1_metrics(all_tp, all_fp, all_fn)
        avg_f1 = all_f1 / len(self.uttids)
        print(f"corpus-level f1 is {corpus_f1:.2f}")
        print(f"avg-level f1 is {avg_f1:.2f}")

    def process_topdown_spans(self, my_dict, uttid): 
        span_levels = self.partition_top_down_spans(my_dict) 
        greedy_extended_span_levels = self.greedy_extend_span_levels(span_levels)
        reverse_greedy_extended_span_levels = self.reverse_greedy_extend_span_levels(greedy_extended_span_levels, uttid)
        return reverse_greedy_extended_span_levels
    
    def process_bottomup_spans(self, my_dict, uttid):
        oracle_phone_seg = self.utt2phnseg[uttid]
        span_levels = {}
        tmp_phone_seg = [(y,z) for (x,y,z) in oracle_phone_seg]
        for step, merge_tuple in my_dict.items():
            first_phn = tmp_phone_seg[merge_tuple[0]]
            second_phn = tmp_phone_seg[merge_tuple[1]]
            tmp_phone_seg[merge_tuple[0]] = (first_phn[0], second_phn[1])
            del tmp_phone_seg[merge_tuple[1]]
            span_levels[step] = copy.copy(tmp_phone_seg)
        
        return span_levels 
     
    def partition_top_down_spans(self, span_list): 
        """ partition a list of spans into different levels 
        each level partition either has multiples of 2 spans, 
        e.g. 0, 2, 4, ...
        
        This function will be useful for plotting spans on top of the spectrograms
        """
        total_spans = len(span_list)
        out = {}
        level = 0 
        root = span_list.pop(0)
        out[level] = [root]
        level += 1
       
        while len(span_list) > 0: 
            span1 = span_list.pop(0)
            span2 = span_list.pop(0)
            current_level_spans = [span1, span2]
            curr_end = max(span1[1], span2[1])
            
            while current_level_spans != []:
    #             print(out)
    #             print(span_list)
    #             print(current_level_spans)
                # take a peak to see if we reach the end 
                if len(span_list) > 0 and curr_end > span_list[0][0]: 
                    out[level] = current_level_spans
                    level += 1
                    current_level_spans = []
                # if not, continue popping 
                elif len(span_list) > 0 and curr_end <= span_list[0][0]:
                    span1 = span_list.pop(0)
                    span2 = span_list.pop(0)
                    current_level_spans.extend([span1, span2])
                    curr_end = max(span1[1], span2[1])
                else: 
                    # edge case: span_list is empty but there are still 
                    # items in current_level_spans
                    out[level] = current_level_spans
                    current_level_spans = []
                    
        assert check_total_dict_items(out) == total_spans
        
        return out 

    def greedy_extend_span_levels(self, span_levels):
        """ 
        for top-down spans:
        greedily create a *complete* span list for each level. If a span between (idx 1, idx 2) is missing, 
        grab it the span from the *previous* level 
        """
        min_v,  max_v = span_levels[0][0][0], span_levels[0][0][1]
        new_span_levels = {}
        for level, span_list in list(span_levels.items()): 
            if level == 0: 
                new_span_levels[level] = span_levels[level]
                continue
            #print(level)
            prev_v = min_v
            new_span_list = []
            for pointer in range(len(span_list)): 
                cur_v = span_list[pointer][0]
                #print(prev_v, cur_v)
                if cur_v != prev_v: 
                    # grab the spans from the above level to complete the list 
                    prev_level_span_list = new_span_levels[level-1]
                    #print(prev_level_span_list)
                    start_idx, end_idx = find_indices_in_list(prev_level_span_list, prev_v, cur_v)
                    #print(start_idx, end_idx)
                    for prev_span_list in prev_level_span_list[start_idx:end_idx+1]:
                        new_span_list.append(prev_span_list)
                new_span_list.append(span_list[pointer])
                prev_v = span_list[pointer][1]
     
            if prev_v != max_v: 
                # grab the spans from the above level to complete the list
                prev_level_span_list = new_span_levels[level-1]
                start_idx, end_idx = find_indices_in_list(prev_level_span_list, prev_v, max_v)
                for prev_span_list in prev_level_span_list[start_idx:end_idx+1]:
                    new_span_list.append(prev_span_list)
            #print(level)
            #print(new_span_list)
            new_span_levels[level] = new_span_list
        return new_span_levels

    def reverse_greedy_extend_span_levels(self, extended_span_levels, uttid):
        """ 
        for top-down spans:
        find the best possible span at each level w.r.t word seg F1. 
        """
        best_word_seg_span_levels = {}
        max_level = len(extended_span_levels.keys())-1
        for level, span_list in reversed(list(extended_span_levels.items())):
            if level == max_level: 
                best_word_seg_span_levels[level] = span_list
                continue 
            new_span_list = copy.copy(span_list)
            curr_span_f1, _ = self.buckeye_word_seg_f1(span_list, uttid, 2)
            for span in new_span_list: 
                next_level_span_list = best_word_seg_span_levels[level+1]
                start_idx, end_idx = find_indices_in_list(next_level_span_list, span[0], span[1])
                if end_idx != start_idx: 
                    # there is split, let's see if the added split can increase the word seg F1 
                    # seek it from *next* level span 
                    added_span = next_level_span_list[start_idx: end_idx+1]                
                    tmp_span_list = copy.copy(new_span_list)
                    i = new_span_list.index(span)
                    if len(added_span) == 1: 
                        tmp_span_list[i] = added_span[0]
                    else: 
                        tmp_span_list[i] = added_span[0]
                        for added_span_item in added_span[1:]:
                            i = i + 1
                            tmp_span_list.insert(i, added_span_item)
                    tmp_span_f1, _ = self.buckeye_word_seg_f1(tmp_span_list, uttid, 2)
                    if tmp_span_f1 > curr_span_f1: 
                        curr_span_f1 = tmp_span_f1
                        new_span_list = tmp_span_list

            best_word_seg_span_levels[level] = new_span_list
            
        return best_word_seg_span_levels 


    def cal_chatgpt_word_seg_f1(self, predicted_segs, uttid):
        """
        ChatGPT's word segmentation F1. Based on segment overlap, which inflats the actual segmentation F1. 
        """
        oracle_word_seg = self.utt2wrdseg[uttid]

        # Convert the oracle and predicted segmentations to sets of word intervals in frames
        oracle_words = set((seg[0], int(seg[1]*self.frame_rate), int(seg[2]*self.frame_rate)) for seg in oracle_word_seg)
        predicted_words = set(('shit', int(seg[0]*self.frame_rate), int(seg[1]*self.frame_rate)) for seg in predicted_segs)

        # Match predicted words to oracle words with 2 frames of tolerance
        tp, fp, fn = 0, 0, 0
        for pred_word in predicted_words:
            best_overlap, best_oracle_word = 0, None
            for oracle_word in oracle_words:
                overlap = max(0, min(oracle_word[2], pred_word[2]) - max(oracle_word[1], pred_word[1]))
                if overlap > best_overlap and overlap >= -1*self.frame_rate: # 20ms seconds tolerance
                    best_overlap = overlap
                    best_oracle_word = oracle_word
            if best_oracle_word is None:
                fp += 1
            else:
                tp += 1
                oracle_words.remove(best_oracle_word)

        fn = len(oracle_words)
        f1_score = f1_metrics(tp, fp, fn)

        return f1_score, (tp, fp, fn)

    def buckeye_word_seg_f1(self, predicted_segs, uttid, tolerance=2):
        """
        Calculate precision, recall, F-score for the segmentation boundaries.
        Parameters
        ----------
        ref : list of vector of bool
            The ground truth reference.
        seg : list of vector of bool
            The segmentation hypothesis.
        tolerance : int
            The number of slices with which a boundary might differ but still be
            regarded as correct.
        Return
        ------
        output : (float, float, float)
            Precision, recall, F-score.
        """
        oracle_word_seg = self.utt2wrdseg[uttid]
        oracle_words = list((int(seg[1]*self.frame_rate), int(seg[2]*self.frame_rate)) for seg in oracle_word_seg)
        oracle_words_seg_bool = create_bool_array(oracle_words)
        ref = [oracle_words_seg_bool]

        predicted_words = list((int(seg[0]*self.frame_rate), int(seg[1]*self.frame_rate)) for seg in predicted_segs)
        predicted_words_seg_bool = create_bool_array(predicted_words)
        seg = [predicted_words_seg_bool]

        n_boundaries_ref = 0
        n_boundaries_seg = 0
        n_boundaries_correct = 0
        for i_boundary, boundary_ref in enumerate(ref):
            boundary_seg = seg[i_boundary]
            assert boundary_ref[-1]  # check if last boundary is True
            assert boundary_seg[-1]
            
            # If lengths are the same, disregard last True reference boundary
            if len(boundary_ref) == len(boundary_seg):
                boundary_ref = boundary_ref[:-1]
                # boundary_seg = boundary_seg[:-1]

            boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                                 # don't want to count this

            # If reference is longer, truncate
            if len(boundary_ref) > len(boundary_seg):
                boundary_ref = boundary_ref[:len(boundary_seg)]
            
            boundary_ref = list(np.nonzero(boundary_ref)[0])
            boundary_seg = list(np.nonzero(boundary_seg)[0])
            n_boundaries_ref += len(boundary_ref)
            n_boundaries_seg += len(boundary_seg)

            for i_seg in boundary_seg:
                for i, i_ref in enumerate(boundary_ref):
                    if abs(i_seg - i_ref) <= tolerance:
                        n_boundaries_correct += 1
                        boundary_ref.pop(i)
                        break

        tp = float(n_boundaries_correct)
        fp = n_boundaries_seg - tp 
        fn = n_boundaries_ref - tp 
        f1_score = f1_metrics(tp, fp, fn)

        return f1_score, (tp, fp, fn)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Hubert Info Align word seg F1 analysis")
    parser.add_argument("--min-pmi", type=int, 
                        choices=[10, 5, 0, -5, -10, -15, -20], 
                        default=-5)
    parser.add_argument("--km-dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es")
    parser.add_argument("--hubert-info-align-ckpt", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/s2u_en.v03.pretrainedmHubert.6LuDecoder.100k.lr5e-4/checkpoints/checkpoint_best.pt")
    parser.add_argument("--parse-out-dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_infoalign_parse/s2u_en-es/spans")
    parser.add_argument("--mfa-dir", type=str,
                        default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/s2u_en-es/')
    parser.add_argument("--split", type=str, default="en-valid_vp-subset100")
    parser.add_argument("--model-version", type=str, 
                        choices=["v02", "v03"], 
                        default="v03")
    parser.add_argument("--parse-alg", type=str, 
                        choices=["top_down", "bottom_up"], 
                        default="bottom_up")

    args = parser.parse_args()

    word_seg_f1 = WordSegF1(args)


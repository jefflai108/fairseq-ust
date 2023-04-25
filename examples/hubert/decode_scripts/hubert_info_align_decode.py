import os, logging, sys 

import soundfile as sf
import copy 
import pickle 
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import fairseq
#from fairseq.examples.hubert.simple_kmeans.feature_utils import dump_feature, get_path_iterator
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data import Dictionary, HubertDataset
from fairseq.tasks.hubert_info_align_pretraining import LabelEncoder

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fairseq.examples.hubert.hubert_info_align_decode')

def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)

def main(
    km_dir, 
    eval_split, 
    hubert_info_align_ckpt, 
    parse_out_dir, 
):
    reset_logging()
    reader = HubertInfoAlignParser(km_dir, eval_split, hubert_info_align_ckpt, parse_out_dir, True, 320000, 100, verbose=True)
    reader.parse()
    #generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    #dump_feature(reader, generator, num, split, nshard, rank, feat_dir)
   
class HubertInfoAlignParser(object):
    def __init__(self, km_dir, eval_split, ckpt_path, parse_out_dir, greedy_decode=True, max_chunk=1600000, parse_window=None, verbose=False):
        self.km_dir = km_dir
        self.eval_split = eval_split
        self.parse_out_dir = parse_out_dir
        self.greedy_decode = greedy_decode
        self.max_chunk = max_chunk      
        self.parse_window = parse_window 
        self.verbose_lev1 = verbose
        self.verbose_lev2 = False

        self.get_uttid_list()

        # setup model and cfg 
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda() 
        self.model.mask_random = False 
        self.model.hide_random = False
        self.task = task
        self.cfg = cfg
        self.label = self.cfg.task.labels[0]
        self.feat2lab_ratio = int(self.cfg.task.sample_rate / self.cfg.task.label_rate)
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f"max_chunk = {self.max_chunk}")

        # parsing parameters 
        self.minimal_span_len = 5 # min is 2 
        self.minimal_splitting_pmi = 0

        # setup dataloader 
        self.set_dictionaries()
        self.load_dataset()
        self.dataloader = DataLoader(self.datasets, batch_size=1, shuffle=False, num_workers=2)

    def get_uttid_list(self): 
        import csv 

        # Define list of extensions to strip
        extensions = ['.wav', '.flac'][0]

       # Open CSV file and read first column
        with open(os.path.join(self.km_dir, f"{self.eval_split}.tsv"), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            filenames = [row[0].split('\t')[0].split('/')[-1][:-len(extensions)] for row in reader]
            
        self.uttids = filenames

    def set_dictionaries(self) -> None:
        self.tgt_dict = Dictionary.load(f"{self.km_dir}/dict.{self.label}.txt")

    def load_dataset(self) -> None:
        manifest = f"{self.km_dir}/{self.eval_split}.tsv"
        dicts = [self.tgt_dict]
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.km_dir}/{self.eval_split}.{self.label}"]

        self.datasets = HubertDataset(
            manifest,
            sample_rate=self.cfg.task.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.task.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=None,
            min_keep_sample_size=None,
            max_sample_size=self.max_chunk,
            shuffle=False, 
            pad_audio=False,
            normalize=self.cfg.task.normalize,
            store_labels=False,
            random_crop=False,
            single_target=self.cfg.task.single_target,
        )

    def parse(self):
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                id = batch["id"].item()
                x = batch["source"].float().cuda() # torch.Size([1, 159360])
                target = batch["label_list"][0].cuda() # torch.Size([1, 497])

                # pick the middle chunk if utt exceeds self.max_chunk
                #offset = 0
                #if x.shape[1] > self.max_chunk: 
                #    offset = int((x.shape[1] - self.max_chunk) / 2)
                #self.max_chunk = 32000
                #x = x[:, offset : offset + self.max_chunk] 
                #target = target[:, int(offset / self.feat2lab_ratio) : int((offset + self.max_chunk) / self.feat2lab_ratio)]

                #assert abs(x.shape[1] / self.feat2lab_ratio - target.shape[1]) <= 1
                #assert x.shape[1] <= self.max_chunk

                # construct prev_output_tokens from targets for teacher-forcing 
                # note that we also do teacher-forcing during inference, as we only 
                # need the log-likelihood. 
                prev_output_tokens = torch.full_like(target, self.tgt_dict.pad()) 
                prev_output_tokens[:, 0].fill_(self.tgt_dict.eos())
                prev_output_tokens[:, 1:] = copy.deepcopy(target[:, :-1]).to(x)

                padding_mask = HubertInfoAlignParser.generate_padding_mask(x)
                mask_span_selected=[(80, 120)]  # inclusive 40

                print(x.shape)
                print(prev_output_tokens.shape)
                print(target.shape)
                print(padding_mask.shape)
          
                # calculate [MASK] sequence logprob 
                #for hide_span_selected in [[(0, 0)], [(0, 1)], [(1, 1)], [(20, 39)], [(20, 39), (140, 150), (160, 200), (350, 360)], [(40, 79), (140, 150), (160, 200), (350, 360)], [(0, 79), (121, 400)]]:
                #    self.parse_single_instance(x, prev_output_tokens, target, padding_mask, mask_span_selected, hide_span_selected)

                # parse via sliding 2-sec window. Run top-down greedy parse within each window. 
                if self.parse_window is None: 
                    self.parse_window = target.shape[1]
                top_down_spans_dict = {}
                for i in range(target.shape[1] // self.parse_window + 1): 
                    curr_top_down_spans = self.top_down_parse_greedy(
                                                x[:, i * self.parse_window * self.feat2lab_ratio: (i+1) * self.parse_window * self.feat2lab_ratio], 
                                                prev_output_tokens[:, i * self.parse_window : (i+1) * self.parse_window], 
                                                target[:, i * self.parse_window : (i+1) * self.parse_window],
                                                padding_mask[:, i * self.parse_window * self.feat2lab_ratio: (i+1) * self.parse_window * self.feat2lab_ratio]
                                          )
                    idx_offset = i * self.parse_window 
                    curr_top_down_spans = [((idx_offset+x,idx_offset+y),score) for ((x,y),score) in curr_top_down_spans]
                    top_down_spans_dict[i] = curr_top_down_spans

                if self.verbose_lev1:
                    logger.info(f"Final top_down_spans: {top_down_spans_dict}")

                self.pickle_spans(top_down_spans_dict, self.uttids[id])

    def pickle_spans(self, spans, uttid): 
        pth = os.path.join(self.parse_out_dir, f"{self.eval_split}-utt{uttid}-minspan{self.minimal_span_len}_minsplitpmi{self.minimal_splitting_pmi}.pkl")
        with open(pth, "wb") as file:
            pickle.dump(spans, file)

    @staticmethod
    def generate_padding_mask(x): 
        padding_mask = torch.BoolTensor(x.shape).fill_(False).to(x)
        return padding_mask

    # parses an utterance top-down by repeatedly splitting the into 
    # spans (i, j), (j, k) to maximize -PMI(i:j, j:k | x)
    def top_down_parse_greedy(self, x, prev_output_tokens, target, padding_mask): 
        top_down_spans = []
        remaining_spans = [((0, target.shape[1]-1), 0)] # end-point inclusive
        while len(remaining_spans) > 0: 
            #(ss, se), curr_span_score = pop_longest(remaining_spans) # priority to longest span --> unnecessary, and this messes up the order
            (ss, se), curr_span_score = remaining_spans.pop(0) # pop the top of the stack
            if self.verbose_lev1:
                logger.info(f"Current popped span is {(ss, se)}")
                logger.info(f"Remaining_spans: {remaining_spans}")
                logger.info(f"Current top_down_spans: {top_down_spans}")
            curr_best_score = -np.inf
            curr_best_split = None 
            for sp in range(ss+self.minimal_span_len, se, self.minimal_span_len): # additional condition to minimize span splitting 
                if self.verbose_lev2:
                    logger.info(f"sp-ss is {sp-ss} and se-sp is {se-sp}")
                if abs(se - sp) < ss+self.minimal_span_len: 
                    break 
                score_xs = self.score_mono(x, prev_output_tokens, target, padding_mask, ss, sp, se)
                if self.verbose_lev1:
                    logger.info(f"score_xs is {score_xs} and split is {sp}")
                if score_xs > curr_best_score: 
                    curr_best_score = score_xs
                    curr_best_split = sp 

            if self.verbose_lev1:
                logger.info(f"best curr_best_score is {curr_best_score} and curr_best_split is {curr_best_split}")
            top_down_spans.append(((ss, se), curr_span_score))
            if curr_best_split is not None: 
                if curr_best_score >= self.minimal_splitting_pmi: # additional condition to stop span splitting. 
                    remaining_spans.append(((ss, curr_best_split-1), curr_best_score.item()))
                    remaining_spans.append(((curr_best_split, se), curr_best_score.item()))
        return top_down_spans

    # computes (len-norm) conditional pointwise mutual information 
    # between the monolingual spans (s, p) and (p, e) (conditioned on the rest of the sequence)
    def score_mono(self, x, prev_output_tokens, target, padding_mask, s, p, e):
        # logp(LEFT) -- [MASK]: s to (p-1); [HIDE]: p to e
        left_mask_span_selected = [(s, p-1)]
        left_hide_span_selected = [(p, e)]
        if self.verbose_lev2: 
            logger.info("logp(LEFT) selections")
            logger.info(left_mask_span_selected)
            logger.info(left_hide_span_selected)
        left_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, left_mask_span_selected, left_hide_span_selected) 
        left_logprob_norm = left_logprob - np.log(p - s)
        
        # logp(RIGHT) -- [HIDE]: s to (p-1); [MASK]: p to e
        right_mask_span_selected = [(p, e)]
        right_hide_span_selected = [(s, p-1)]
        if self.verbose_lev2: 
            logger.info("logp(RIGHT) selections")
            logger.info(right_mask_span_selected)
            logger.info(right_hide_span_selected)
        right_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, right_mask_span_selected, right_hide_span_selected)
        right_logprob_norm = right_logprob - np.log(e - p + 1)

        # logp(JOINT) -- [MASK]: s to e 
        joint_mask_span_selected = [(s, e)]
        joint_hide_span_selected = [(0, 0)]
        if self.verbose_lev2:
            logger.info("logp(JOINT) selections")
            logger.info(joint_mask_span_selected)
            logger.info(joint_hide_span_selected)
        joint_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, joint_mask_span_selected, joint_hide_span_selected) 
        joint_logprob_norm = joint_logprob - np.log(e - s + 1)

        # PMI = logp(JOINT) - logp(LEFT) - logp(RIGHT)
        pmi = (joint_logprob_norm - left_logprob_norm - right_logprob_norm) 

        # minimize PMI = maximize -PMI
        score = -pmi 
        return score 

    def parse_single_instance(self, x, prev_output_tokens, target, padding_mask, mask_span_selected, hide_span_selected): 
        """ return logprob(x1 | x - [x1: MASK] - [x2: HIDE]) """

        net_output = self.model(
                        source=x, 
                        prev_output_tokens=prev_output_tokens, 
                        target_list=[target], 
                        padding_mask=padding_mask, 
                        mask=True, 
                        output_layer=None, 
                        mask_span_selected=mask_span_selected,  
                        hide_span_selected=hide_span_selected, 
                    )
        
        logp_m = self.model.get_logits(net_output)[0]
        targ_m = self.model.get_targets(net_output)[0]
       
        # logprob of target sequence under [MASK]
        targ_seq_logp = logp_m[np.arange(len(targ_m)), targ_m]
        targ_seq_logp_sum = targ_seq_logp.sum()

        # [MASK] pred acc., and logprob of greedy pred sequence under [MASK]
        masked_pred_acc, greedy_pred_seq = self.single_prediction_acc(logp_m, targ_m)
        if self.verbose_lev2: 
            logger.info('[MASK] token prediction accuracy is %.2f' % masked_pred_acc)
        greedy_pred_seq_logp = logp_m[np.arange(len(greedy_pred_seq)), greedy_pred_seq]

        return targ_seq_logp_sum 
    
    def single_prediction_acc(self, logp, targ): 
        targ = targ.to(logp.device)
        if self.greedy_decode: # argmax 
            pred_seq = torch.argmax(logp, dim=1)
        else: # sample from distribution 
            pred_seq = torch.multinomial(torch.exp(logp), num_samples=1).squeeze()

        corr  = (pred_seq == targ).sum().item()
        count = len(targ)
        acc = (corr / count)

        return acc, pred_seq

def pop_longest(stack):
    stack.sort(key=lambda t: abs(t[0][1] - t[0][0]), reverse=False)
    return stack.pop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Hubert Info Align decoding")
    parser.add_argument("--km-dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es")
    parser.add_argument("--hubert-info-align-ckpt", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/s2u_en.v02.pretrainedmHubert.6LuDecoder.200k.lr5e-4/checkpoints/checkpoint_best.pt")
    parser.add_argument("--parse-out-dir", type=str, 
                        default="/data/sls/scratch/clai24/lexicon/exp/hubert_infoalign_parse/s2u_en-es/spans")
    parser.add_argument("--split", type=str, default="en-test_epst")

    args = parser.parse_args()

    main(args.km_dir, args.split, args.hubert_info_align_ckpt, args.parse_out_dir)

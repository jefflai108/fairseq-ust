import os, logging, sys 
import soundfile as sf
import copy 
import pickle 
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import fairseq
from fairseq.data import Dictionary, HubertDataset
from fairseq.tasks.hubert_info_align_pretraining import LabelEncoder

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fairseq.examples.hubert.hubert_info_align_phn_decode')

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
    mfa_dir, 
    eval_split, 
    model_version, 
    parse_alg, 
    hubert_info_align_ckpt, 
    parse_out_dir, 
    min_pmi, 
    njobs, 
    job_id, 
):
    reset_logging()
    reader = HubertInfoAlignParser(km_dir, mfa_dir, eval_split, model_version, parse_alg, hubert_info_align_ckpt, parse_out_dir, min_pmi, njobs, job_id, True, verbose=True)
    reader.parse()
   
class HubertInfoAlignParser(object):
    def __init__(self, km_dir, mfa_dir, eval_split, model_version, parse_alg, ckpt_path, parse_out_dir, min_pmi, njobs, job_id, greedy_sample=True, verbose=False):
        # setup arguments 
        self.km_dir = km_dir
        self.mfa_dir = mfa_dir 
        self.eval_split = eval_split
        self.model_version = model_version
        self.parse_alg = parse_alg
        if parse_alg == "bottom_up":
            self.minimal_merging_pmi = min_pmi
        elif parse_alg == "top_down": 
            self.minimal_splitting_pmi = min_pmi 
        self.attention_context_window = 512 # this is needed because there is encoder cross-attention in our decoder, and therefore we want to avoid the [MASK] region to be >1024 tokens. 
                                            # Because we have [MASK] and [HIDE], we restrict each to be <512 tokens. We therefore introduce self.attention_context_window to trim [MASK] and [HIDE] spans. 
        self.parse_out_dir = parse_out_dir
        self.njobs = njobs
        self.job_id = job_id
        self.greedy_sample = greedy_sample
        self.verbose_lev1 = verbose
        self.verbose_lev2 = False

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

        # setup data
        self.get_uttid_list()
        self.read_segmentation_pickle()

        # setup dataloader 
        self.set_dictionaries()
        self.load_dataset()
        self.dataloader = DataLoader(self.datasets, batch_size=1, shuffle=False, num_workers=2)

    def get_uttid_list(self): 
        import csv 

       # Open CSV file and read first column
        with open(os.path.join(self.km_dir, f"{self.eval_split}.tsv"), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            filenames = [row[0].split('\t')[0].split('/')[-1] for row in reader]

            self.utt_id_offset = 0
            if self.njobs > 1: 
                interv = len(filenames) // self.njobs
                filenames = filenames[interv * self.job_id: interv * (self.job_id + 1)]
                self.utt_id_offset = interv * self.job_id
                if len(filenames) == 0: 
                    logger.info('no filenames')
                    exit()

            trimmed_filenames = filenames
            len_extensions = 0
            if filenames[0].endswith('.flac'): 
                len_extensions = len('.flac')
            elif filenames[0].endswith('.wav'):
                len_extensions = len('.wav')
            
            if len_extensions > 0: 
                trimmed_filenames = [filename[:-len_extensions] for filename in filenames]
            
        self.uttids = trimmed_filenames
        self.total_utt_cnt = len(self.uttids)
        logger.info(f'total number of utt is {self.total_utt_cnt}')

    def read_segmentation_pickle(self): 
        with open(os.path.join(args.mfa_dir, f"{args.split}-phones_seg.pkl"), "rb") as file:
            self.utt2seg = pickle.load(file)

        self.utt2seg_frames, self.utt2seg_wavs = {}, {}
        for utt, seg_list in self.utt2seg.items(): 
            seg_list_in_frames = [(int(y*self.cfg.task.label_rate),int(z*self.cfg.task.label_rate)) for (x,y,z) in seg_list]
            seg_list_in_wav = [(int(y*self.cfg.task.label_rate*self.feat2lab_ratio),int(z*self.cfg.task.label_rate*self.feat2lab_ratio)) for (x,y,z) in seg_list]
            self.utt2seg_frames[utt] = seg_list_in_frames
            self.utt2seg_wavs[utt] = seg_list_in_wav

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
            max_sample_size=None,
            shuffle=False, 
            pad_audio=False,
            normalize=self.cfg.task.normalize,
            store_labels=False,
            random_crop=False,
            single_target=self.cfg.task.single_target,
        )

    def parse(self):
        with torch.no_grad():
            utt_cnt = 0
            for batch in tqdm(self.dataloader):
                if batch["id"].item() < self.utt_id_offset:
                    continue 
                id = batch["id"].item() - self.utt_id_offset
                if utt_cnt >= self.total_utt_cnt: 
                    break
                utt_cnt += 1
                x = batch["source"].float().cuda()
                target = batch["label_list"][0].cuda() 

                # construct prev_output_tokens from targets for teacher-forcing 
                # note that we also do teacher-forcing during inference, as we only 
                # need the log-likelihood. 
                prev_output_tokens = torch.full_like(target, self.tgt_dict.pad()) 
                prev_output_tokens[:, 0].fill_(self.tgt_dict.eos())
                prev_output_tokens[:, 1:] = copy.deepcopy(target[:, :-1]).to(x)

                padding_mask = HubertInfoAlignParser.generate_padding_mask(x)

                if self.verbose_lev1:
                    logger.info(x.shape)
                    logger.info(prev_output_tokens.shape)
                    logger.info(target.shape)
                    logger.info(padding_mask.shape)

                seg_frames = self.utt2seg_frames[self.uttids[id]]
                # trim seg_frames based on len(target)
                if seg_frames[-1][0] >= target.shape[1]-1: 
                    del seg_frames[-1]
                else:
                    seg_frames[-1] = (seg_frames[-1][0], target.shape[1]-1)

                if self.verbose_lev1:
                    logger.info(len(seg_frames))
                    logger.info(seg_frames)
    
                if self.parse_alg == "bottom_up":
                    # bottom-up. Each round only merge two consecutive segments, based on highest PMI. 
                    total_num_parse_steps = len(seg_frames) - 1
                    bottom_up_merge_dict = {}
                    for i in range(total_num_parse_steps): 
                        curr_best_merge, seg_frames = self.bottom_up_parse_greedy(x, prev_output_tokens, target, padding_mask, seg_frames)
                        if curr_best_merge is None: 
                            break 
                        bottom_up_merge_dict[i] = curr_best_merge
                        if self.verbose_lev1:
                            logger.info(f"bottom up parse step {i} and bottom_up_merge_dict is {bottom_up_merge_dict}")

                    if self.verbose_lev1:
                        logger.info(f"Final bottom_up_merge_dict is {bottom_up_merge_dict}")
                    self.pickle_spans(bottom_up_merge_dict, self.uttids[id])

                elif self.parse_alg == "top_down":
                    top_down_spans = self.top_down_parse_greedy(x, prev_output_tokens, target, padding_mask, seg_frames)

                    if self.verbose_lev1:
                        logger.info(f"Final top_down_spans: {top_down_spans}")
                    self.pickle_spans(top_down_spans, self.uttids[id])

    def pickle_spans(self, spans, uttid): 
        if self.parse_alg == "bottom_up":
            pth = os.path.join(self.parse_out_dir, f"{uttid}-minmergepmi{self.minimal_merging_pmi}.pkl")
        elif self.parse_alg == "top_down": 
            pth = os.path.join(self.parse_out_dir, f"{uttid}-minsplitpmi{self.minimal_splitting_pmi}.pkl")
        with open(pth, "wb") as file:
            pickle.dump(spans, file)

    @staticmethod
    def generate_padding_mask(x): 
        padding_mask = torch.BoolTensor(x.shape).fill_(False).to(x)
        return padding_mask

    # parses an utterance bottom-up by repeatedly merging consecutive 
    # spans (i, j), (j, k) to maximize PMI(i:j, j:k | x)
    def bottom_up_parse_greedy(self, x, prev_output_tokens, target, padding_mask, seg_frames): 
        curr_best_score = -np.inf
        curr_best_merge = None 
        for i in range(0, len(seg_frames) - 1):
            if self.verbose_lev1:
                logger.info(f"selected seg_frames idx is {i}, {i+1} out of {len(seg_frames)}")
            first_seg_frame, second_seg_frame = seg_frames[i], seg_frames[i+1]

            # if either MASK or HIDE exceeds >512 frames, we ignore such possible merge
            if first_seg_frame[1] - first_seg_frame[0] > self.attention_context_window:
                continue 
            if second_seg_frame[1] - second_seg_frame[0] > self.attention_context_window:
                continue

            score_xs = self.score_mono(x, prev_output_tokens, target, padding_mask, (first_seg_frame, second_seg_frame))
            if score_xs >= self.minimal_merging_pmi and score_xs > curr_best_score: 
                curr_best_score = score_xs
                curr_best_merge = (i, i+1)
            if self.verbose_lev1:
                logger.info(f"score_xs is {score_xs} and current_best_merge is is {curr_best_merge}")

        # merge seg_frames
        if curr_best_merge is not None:
            first_merge, second_merge = curr_best_merge
            seg_frames[first_merge] = (seg_frames[first_merge][0], seg_frames[second_merge][1])
            del seg_frames[second_merge]

            if self.verbose_lev1:
                logger.info(f"new seg_frames is {seg_frames}")

        return curr_best_merge, seg_frames

    # parses an utterance top-down by repeatedly splitting the into 
    # spans (i, j), (j, k) to maximize -PMI(i:j, j:k | x)
    def top_down_parse_greedy(self, x, prev_output_tokens, target, padding_mask, seg_frames): 
        top_down_spans = []
        remaining_spans = [((0, len(seg_frames)-1), 0)] # keep track of the start and end indexes of seg_frames. End-point inclusive
        while len(remaining_spans) > 0: 
            (start_seg_idx, end_seg_idx), curr_span_score = remaining_spans.pop(0) # pop the top of the stack
            if self.verbose_lev1:
                logger.info(f"Current popped span is {seg_frames[start_seg_idx: end_seg_idx+1]}")
                logger.info(f"Remaining_spans: {remaining_spans}")
            curr_best_score = -np.inf
            curr_best_split = None 

            for split_seg_idex in range(start_seg_idx, end_seg_idx):
                first_seg_frame = (seg_frames[start_seg_idx][0], seg_frames[split_seg_idex][1])
                second_seg_frame = (seg_frames[split_seg_idex+1][0], seg_frames[end_seg_idx][1])

                # restrict each MASK and HIDE span to be <512 frames (~10 seconds of audio) from the split point 
                if first_seg_frame[1] - first_seg_frame[0] > self.attention_context_window: 
                    first_seg_frame = (first_seg_frame[1]-self.attention_context_window+1, first_seg_frame[1])
                if second_seg_frame[1] - second_seg_frame[0] > self.attention_context_window: 
                    second_seg_frame = (second_seg_frame[0], second_seg_frame[0]+self.attention_context_window-1)

                if self.verbose_lev1:
                    logger.info(f"Current first_seg_frame is {first_seg_frame}")
                    logger.info(f"Current second_seg_frame is {second_seg_frame}")

                score_xs = self.score_mono(x, prev_output_tokens, target, padding_mask, (first_seg_frame, second_seg_frame))
                if self.verbose_lev1:
                    logger.info(f"score_xs is {score_xs} and split is {split_seg_idex}")
                if score_xs > curr_best_score: 
                    curr_best_score = score_xs
                    curr_best_split = split_seg_idex

            if self.verbose_lev1:
                logger.info(f"best curr_best_score is {curr_best_score} and curr_best_split is {curr_best_split}")
            top_down_spans.append(((start_seg_idx, end_seg_idx), curr_span_score)) 
            if curr_best_split is not None: 
                if curr_best_score >= self.minimal_splitting_pmi: # additional condition to stop span splitting. 
                    remaining_spans.append(((start_seg_idx, curr_best_split), curr_best_score.item()))
                    remaining_spans.append(((curr_best_split+1, end_seg_idx), curr_best_score.item()))
        return top_down_spans

    # computes (len-norm) conditional pointwise mutual information 
    # between the monolingual spans (s, p) and (p, e) (conditioned on the rest of the sequence)
    def score_mono(self, x, prev_output_tokens, target, padding_mask, seg_frame_tuple):
        # logp(LEFT) -- [MASK]: s to (p-1); [HIDE]: p to e
        left_mask_span_selected = [seg_frame_tuple[0]]
        left_hide_span_selected = [seg_frame_tuple[1]]
        if self.verbose_lev2: 
            logger.info("logp(LEFT) selections")
            logger.info(left_mask_span_selected)
            logger.info(left_hide_span_selected)
        left_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, left_mask_span_selected, left_hide_span_selected) 
        left_logprob_norm = left_logprob - np.log(seg_frame_tuple[0][1] - seg_frame_tuple[0][0] + 1)
        
        # logp(RIGHT) -- [HIDE]: s to (p-1); [MASK]: p to e
        right_mask_span_selected = [seg_frame_tuple[1]]
        right_hide_span_selected = [seg_frame_tuple[0]]
        if self.verbose_lev2: 
            logger.info("logp(RIGHT) selections")
            logger.info(right_mask_span_selected)
            logger.info(right_hide_span_selected)
        right_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, right_mask_span_selected, right_hide_span_selected)
        right_logprob_norm = right_logprob - np.log(seg_frame_tuple[1][1] - seg_frame_tuple[1][0] + 1)

        # logp(JOINT) -- [MASK]: s to e 
        joint_mask_span_selected = [(seg_frame_tuple[0][0], seg_frame_tuple[1][1])]
        joint_hide_span_selected = [(0, 0)]
        if self.verbose_lev2:
            logger.info("logp(JOINT) selections")
            logger.info(joint_mask_span_selected)
            logger.info(joint_hide_span_selected)
        joint_logprob = self.parse_single_instance(x, prev_output_tokens, target, padding_mask, joint_mask_span_selected, joint_hide_span_selected) 
        joint_logprob_norm = joint_logprob - np.log(seg_frame_tuple[1][1] - seg_frame_tuple[0][0] + 1)

        # PMI = logp(JOINT) - logp(LEFT) - logp(RIGHT)
        pmi = (joint_logprob_norm - left_logprob_norm - right_logprob_norm) 

        # bottom-up: maximize consecutive spans PMI 
        # top-down: minimize consecutive spans PMI 
        if self.parse_alg == "bottom_up": 
            score = pmi 
        elif self.parse_alg == "top_down": 
            score = -pmi 

        return score 

    def parse_single_instance(self, x, prev_output_tokens, target, padding_mask, mask_span_selected, hide_span_selected): 
        """ return logprob(x1 | x - [x1: MASK] - [x2: HIDE]) """
        
        if self.model_version == 'v03':
            dummy_alignment = [[(-1,0)]]
            net_output = self.model(
                            source=x, 
                            prev_output_tokens=prev_output_tokens, 
                            alignment_list=dummy_alignment, 
                            target_list=[target], 
                            padding_mask=padding_mask, 
                            mask=True, 
                            output_layer=None, 
                            mask_span_selected=mask_span_selected,  
                            hide_span_selected=hide_span_selected, 
                        )
        elif self.model_version == 'v02':
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
        if self.model_version == 'v03': # ensure target units are indeed deduplicated
            assert self.check_target_deduplicated(targ_m)

        # logprob of target sequence under [MASK]
        targ_seq_logp = logp_m[np.arange(len(targ_m)), targ_m]
        if self.verbose_lev2:
            logger.info(targ_seq_logp)
            logger.info(len(targ_seq_logp))
        targ_seq_logp_sum = targ_seq_logp.sum()

        # [MASK] pred acc., and logprob of greedy pred sequence under [MASK]
        masked_pred_acc, greedy_pred_seq = self.single_prediction_acc(logp_m, targ_m)
        if self.verbose_lev2: 
            logger.info('[MASK] token prediction accuracy is %.2f' % masked_pred_acc)
        greedy_pred_seq_logp = logp_m[np.arange(len(greedy_pred_seq)), greedy_pred_seq]

        return targ_seq_logp_sum 
    
    def single_prediction_acc(self, logp, targ): 
        targ = targ.to(logp.device)
        if self.greedy_sample: # argmax 
            pred_seq = torch.argmax(logp, dim=1)
        else: # sample from distribution 
            pred_seq = torch.multinomial(torch.exp(logp), num_samples=1).squeeze()

        corr  = (pred_seq == targ).sum().item()
        count = len(targ)
        acc = (corr / count)

        return acc, pred_seq

    def check_target_deduplicated(self, targ_m): 
        deduplicate_targ_m = torch.cat((torch.unique_consecutive(targ_m[:-1]), targ_m[-1:]))
        has_consecutive_deduplicated = torch.equal(targ_m, deduplicate_targ_m)

        return has_consecutive_deduplicated

def pop_longest(stack):
    stack.sort(key=lambda t: abs(t[0][1] - t[0][0]), reverse=False)
    return stack.pop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Hubert Info Align phone-to-word decoding")
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
    parser.add_argument("--njobs", default=1, type=int, 
                        help="number of parallel jobs")
    parser.add_argument("--job-id", default=0, type=int, 
                        help="parallel job id #")

    args = parser.parse_args()
    
    assert args.model_version in args.hubert_info_align_ckpt
    parse_out_dir = os.path.join(args.parse_out_dir, f"{args.model_version}.{args.split}.{args.parse_alg}")
    os.makedirs(parse_out_dir, exist_ok=True)

    main(args.km_dir, args.mfa_dir, args.split, args.model_version, args.parse_alg, args.hubert_info_align_ckpt, parse_out_dir, args.min_pmi, args.njobs, args.job_id)

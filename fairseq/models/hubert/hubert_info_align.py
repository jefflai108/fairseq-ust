# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
#from fairseq.data.data_utils import compute_mask_and_hide_indices
from fairseq.data.info_align_data_utils import (
    compute_mask_and_hide_indices, 
    compute_neighboring_mask_and_hide_indices,
    compute_neighboring_phn_mask_and_phn_hide_indices, 
)
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models.fairseq_model import check_type
from fairseq.models import (
    BaseFairseqModel, 
    FairseqEncoderDecoderModel,
    FairseqDecoder, 
    FairseqEncoder,
    register_model,
)
from fairseq.models.speech_to_speech.modules import StackedEmbedding
from fairseq.models.unit_to_unit.unit_decoder import TransformerUnitDecoder
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    LAYER_TYPE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.models.hubert.hubert import (
    HubertConfig, 
    HubertModel
)
from fairseq.tasks.hubert_info_align_pretraining import (
    HubertInfoAlignPretrainingConfig,
    HubertInfoAlignPretrainingTask,
    HubertInfoAlignPhnMaskPretrainingConfig,
    HubertInfoAlignPhnMaskPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class HubertInfoAlignConfig(HubertConfig):
    # Hubert encoder init ckpt 
    pretrained_hubert_ckpt: str = field(
        default="", 
        metadata={
            "help": "load a pre-trained Hubert model ckpt for info-align pre-training."
            "If none, info-align training from scratch."
        },
    )

    # Hubert encoder masking params
    mask_lengths: List[int] = field(default_factory=list, metadata={"help": "[MASK] mask length"})
    hide_lengths: List[int] = field(default_factory=list, metadata={"help": "[HIDE] mask length"})
    mask_random: bool = field(
        default=True, 
        metadata={"help": "randomly applied [MASK] masks to feature space."}
    )
    hide_random: bool = field(
        default=True, 
        metadata={"help": "randomly applied [HIDE] masks to feature space."}
    )
    num_mask_spans: List[int] = field(
        default_factory=list, 
        metadata={"help": "number of [MASK] spans, where each span is of mask_length"}
    )
    num_hide_spans: List[int] = field(
        default_factory=list, 
        metadata={"help": "number of [HIDE] spans, where each span is of hide_length"}
    )

    # AR unit decoder params
    decoder_encoder_embed_dim: int = field(
        default=256,
        metadata={"help": "decoder embedding dimension"},
    )
    decoder_embed_dim: int = field(
        default=256,
        metadata={"help": "decoder embedding dimension"},
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, 
        metadata={"help": "decoder embedding dimension for FFN"},
    )
    decoder_layers: int = field(
        default=6,
        metadata={"help": "num decoder layers"},
    )
    decoder_attention_heads: int = field(
        default=8, 
        metadata={"help": "num decoder attention heads"},
    )
    decoder_normalize_before: bool = field(
        default=True,
        metadata={"help": "apply layernorm before each decoder block"}, 
    )
    decoder_learned_pos: bool = field(
        default=False, 
    )
    share_decoder_input_output_embed: bool = field(
        default=True, 
    )
    no_token_positional_embeddings: bool = field(
        default=False
    )
    decoder_layerdrop: float = field(
        default=0.0
    )
    decoder_output_dim: int = field(
        default=256
    )
    decoder_input_dim: int = field(
        default=256
    )
    decoder_dropout: float = field(
        default=0.1
    )
    decoder_activation_dropout: float = field(
        default=0.1
    )
    relu_dropout: float = field(
        default=0.1
    )
    adaptive_softmax_dropout: float = field(
        default=0.0
    )
    n_frames_per_step: int = field(
        default=1
    )

@register_model("hubert_info_align", dataclass=HubertInfoAlignConfig)
class HubertInfoAlignModel(BaseFairseqModel):
    def __init__(
        self,
        hubert_encoder: HubertModel, 
        unit_decoder: TransformerUnitDecoder, 
        cfg: HubertInfoAlignConfig,
        task_cfg: HubertInfoAlignPretrainingConfig,
        tgt_dict: Dictionary,
    ) -> None:
        logger.info(f"HubertInfoAlignModel Config: {cfg}")

        super().__init__()

        self.tgt_dict = tgt_dict

        # init encoder and decoder 
        self.encoder = hubert_encoder
        self.decoder = unit_decoder

        check_type(self.encoder, HubertModel)
        check_type(self.decoder, FairseqDecoder)

        self.encoder.train().cuda()
        self.decoder.train().cuda()

        # set CNN feature extractor to non-finetunable
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad_(False)
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        # learnable [MASK] and [HIDE] masks
        self.mask_emb = self.encoder.mask_emb
        self.hide_emb = nn.Parameter(
            torch.FloatTensor(self.mask_emb.size()[0]).uniform_()
        )
        self.hide_emb = self.hide_emb

        # override the pretrained Hubert's masking parameters 
        self.mask_lengths = cfg.mask_lengths
        self.hide_lengths = cfg.hide_lengths
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_random = cfg.mask_random
        self.hide_random = cfg.hide_random
        self.num_mask_spans = cfg.num_mask_spans
        self.num_hide_spans = cfg.num_hide_spans

    @classmethod
    def build_model(cls, cfg: HubertInfoAlignConfig, task: HubertInfoAlignPretrainingTask):
        
        assert len(task.dictionaries) == 1
        tgt_dict = task.dictionaries[0]

        # Hubert encoder 
        encoder = cls.build_encoder(cfg.pretrained_hubert_ckpt, cfg, task.cfg, tgt_dict)

        # Autoregressive unit decoder 
        cfg.encoder_embed_dim = cfg.decoder_encoder_embed_dim
        cfg.dropout = cfg.decoder_dropout
        cfg.activation_dropout = cfg.decoder_activation_dropout
        decoder = cls.build_decoder(cfg, tgt_dict)

        # HubertInfoAlign model 
        base_model = HubertInfoAlignModel(encoder, 
                                          decoder, 
                                          cfg, 
                                          task.cfg, 
                                          tgt_dict)

        return base_model

    @classmethod
    def build_encoder(cls, ckpt_str, cfg, task_cfg, tgt_dict):
        encoder = cls.load_pretrained_hubert_ckpt(ckpt_str, cfg, task_cfg, tgt_dict)
        return encoder

    @classmethod
    def load_pretrained_hubert_ckpt(cls, ckpt_str, cfg, task_cfg, tgt_dict): 
        """load a pretrained Hubert seed model for info-align pretraining"""
        import fairseq.checkpoint_utils, sys 
        if ckpt_str == "":
            logger.info("Initialize from a HubertModel from scratch")
            return HubertModel(cfg, task_cfg, [tgt_dict])
        else:
            logger.info(f"Initialize from a pretrained HubertModel at {ckpt_str}")
            (
                model,
                cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_str])
            return model[0]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_decoder(cls, cfg, tgt_dict):
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            cfg.decoder_embed_dim,
            padding_idx,
            num_stacked=cfg.n_frames_per_step,
        )

        return TransformerUnitDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
        )

    def apply_encoder_mask_and_hide(self, 
                   x, 
                   padding_mask, 
                   mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
                   hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
                  ):
        """compute and apply [MASK] and [HIDE] spans to x. 
        The spans can be sampled at random or pre-specified, and in 
        either way, no two spans overlap. 

        Exmaples of pre-specifying [MASK] and [HIDE] spans:
            mask_span_selected = [(2, 10), (67, 98)]
            hide_span_selected = [(21, 45)]
        """
        
        B, T, C = x.shape
        mask_indices, hide_indices = None, None
        if (len(self.num_mask_spans) > 0 or 
            len(self.hide_mask_spans) > 0 or 
            mask_span_selected != [(0, 0)] or 
            hide_span_selected != [(0, 0)]
            ):
            mask_indices, hide_indices = compute_neighboring_mask_and_hide_indices(
                (B, T),
                padding_mask,
                self.mask_lengths,
                self.hide_lengths,
                mask_random=self.mask_random, 
                hide_random=self.hide_random, 
                mask_span_selected=mask_span_selected, 
                hide_span_selected=hide_span_selected, 
                num_mask_spans=self.num_mask_spans, 
                num_hide_spans=self.num_hide_spans, 
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                mask_dropout=0.0, 
                hide_dropout=0.0, 
            )
            #logger.info(mask_indices[0])
            #logger.info(hide_indices[0])
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            hide_indices = torch.from_numpy(hide_indices).to(x.device)
            x[mask_indices] = self.mask_emb
            x[hide_indices] = self.hide_emb

            return x, mask_indices, hide_indices
        else: 
            return x, None, None 

    def forward(
        self,
        source: torch.Tensor, 
        prev_output_tokens: torch.Tensor, 
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_layer: Optional[int] = None,
        mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
        hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
    ):
        assert target_list is None or len(target_list) == 1
        target = target_list[0]

        # forward encoder 
        encoder_out = self.forward_encoder(
            source, 
            target, 
            padding_mask, 
            mask, 
            output_layer, 
            mask_span_selected, 
            hide_span_selected,
        )
 
        # determine [MASK] regions
        masked_indices = torch.logical_and(~encoder_out["encoder_padding_mask"][0], 
                                            encoder_out["mask_indices"][0])
        bsz = prev_output_tokens.shape[0]
        # determine max length of [MASK] region 
        # this saves space compared to 
        # max_masked_len = self.mask_lengths[-1] * max(self.num_mask_spans)
        max_masked_len = 0
        for i in range(bsz): 
            true_indices = torch.where(masked_indices[i])[0]
            max_masked_len = max(max_masked_len, len(true_indices))
        max_masked_len += max(self.num_mask_spans)-1 # account for the addition of separator token 
        masked_prev_output_tokens = torch.full((bsz, max_masked_len), self.tgt_dict.pad()).to(prev_output_tokens)
        masked_target = torch.full((bsz, max_masked_len), -1).to(prev_output_tokens)

        def extract_masked_consecutive_id(masked_bool, unit_seq): 
            # get indices of all True values
            true_indices = torch.where(masked_bool)[0]
        
            # check if indices form consecutive sequence -- there is only 1 [MASK] region
            if len(true_indices) > 0 and true_indices[-1] - true_indices[0] == len(true_indices) - 1:
                masked_unit_seq = unit_seq[true_indices]
                return masked_unit_seq

            # Below is for cases where there are multiple [MASK] regions. 
            # We want to extract the [MASK] regions separately such that we can insert 
            # a separator token (<eos>) in between them. 

            # initialize variables for sequence tracking
            true_indices_sequences = []
            current_sequence = []
            last_index = None

            # iterate over indices and identify consecutive sequences
            for index in true_indices.tolist():
                if last_index is None:
                    # start new sequence
                    current_sequence.append(index)
                elif index == last_index + 1:
                    # continue current sequence
                    current_sequence.append(index)
                else:
                    # end current sequence and start new sequence
                    true_indices_sequences.append(current_sequence)
                    current_sequence = [index]
                last_index = index

            # append final sequence
            true_indices_sequences.append(current_sequence)

            # filter out sequences of length 1
            true_indices_sequences = [seq for seq in true_indices_sequences if len(seq) > 1]
            true_indices_sequences = [torch.tensor(seq, device=true_indices.device) for seq in true_indices_sequences]
            sequences = [unit_seq[true_idx] for true_idx in true_indices_sequences]

            # insert the separator token in between the [MASK] regions 
            separator = torch.tensor([self.tgt_dict.eos()]).to(true_indices)
            sequences = torch.cat([torch.cat([seq, separator]) for seq in sequences], dim=0)
            return sequences[:-1]

        for i in range(bsz): 
            masked_bool = masked_indices[i]
            #logger.info(masked_bool)
            masked_unit_seq = extract_masked_consecutive_id(masked_bool, prev_output_tokens[i])
            masked_target_seq = extract_masked_consecutive_id(masked_bool, encoder_out["target"][i])
            assert len(masked_unit_seq) == len(masked_target_seq)
            masked_prev_output_tokens[i][: len(masked_unit_seq)] = masked_unit_seq
            masked_target[i][: len(masked_target_seq)] = masked_target_seq
        #logger.info(masked_prev_output_tokens)
        #logger.info(masked_target)

        # forward decoder 
        # IMPORTANT: only feed the decoder the [MASK] unit seq instead of the full unit seq!
        decoder_out, _ = self.decoder(
            masked_prev_output_tokens,
            encoder_out=encoder_out,
        )
        #logger.info(decoder_out.shape) # torch.Size([4, 50, 1004])
        #logger.info(masked_prev_output_tokens.shape) # torch.Size([4, 50])
        #logger.info(masked_target.shape) # torch.Size([4, 50])
        #logger.info(masked_target)

        # compute logprob on [MASK] regions
        logit_m = decoder_out[masked_prev_output_tokens != self.tgt_dict.pad()]
        #logger.info(logit_m.shape) # torch.Size([133, 1004])

        target_list_m = masked_target[masked_target != -1]
        #logger.info(target_list_m.shape) # torch.Size([133])
        #logger.info(target_list_m)

        result = {
            "logit_m_list": [logit_m],
            "target_list": [target_list_m], 
        }
        return result

    def forward_encoder_features(self, source: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): 
            features = self.encoder.feature_extractor(source)
            
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        features, target_list = self.encoder.forward_targets(features, [target])

        return features, target_list[0]

    def forward_encoder_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        padding_mask = self.encoder.forward_padding_mask(features, padding_mask)

        return padding_mask

    def forward_encoder(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_layer: Optional[int] = None,
        mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
        hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
    ) -> Dict[str, torch.Tensor]:

        #logger.info(source.shape) # torch.Size([4, 67200])
        #logger.info(target.shape) # torch.Size([4, 209])
        #logger.info(padding_mask.shape)
        features = self.forward_encoder_features(source)
        #logger.info(features.shape) # torch.Size([4, 512, 209])
        if target is not None:
            features, target = self.forward_targets(features, target)

        #logger.info(features.shape) # torch.Size([4, 512, 209])
        #logger.info(target.shape) # torch.Size([4, 209])
        #logger.info(padding_mask)
        #logger.info(padding_mask.shape) # torch.Size([4, 67200])
        #logger.info(mask) # True 
        #logger.info(output_layer) # None

        features = features.transpose(1, 2) # torch.Size([4, 209, 512])
        features = self.encoder.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_encoder_padding_mask(features, padding_mask)

        #logger.info(padding_mask.shape) # torch.Size([4, 209])
        #logger.info(padding_mask)

        if self.encoder.post_extract_proj is not None:
            features = self.encoder.post_extract_proj(features)
        #logger.info(features.shape) # torch.Size([4, 209, 768])

        features = self.encoder.dropout_input(features)


        if mask:
            x, mask_indices, hide_indices = self.apply_encoder_mask_and_hide(features, padding_mask, mask_span_selected, hide_span_selected)
        else:
            x = features
            mask_indices = None
            hide_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        
        #logger.info(x.shape) # torch.Size([4, 209, 768])
        proj_x = self.encoder.final_proj(x) # D: 768 --> 256
        proj_x = proj_x.transpose(0, 1)
        features = features.transpose(0, 1)

        #logger.info(proj_x.shape) # torch.Size([209, 4, 256])
        #logger.info(features.shape) # torch.Size([209, 4, 768])
        
        return {
                "encoder_out": [proj_x],  # T x B x C
                "encoder_padding_mask": [padding_mask] if padding_mask is not None else [],  # B x T
                "mask_indices": [mask_indices] if mask_indices is not None else [], # B x T
                "hide_indices": [hide_indices] if hide_indices is not None else [], # B x T
                "features": features, # T x B x C
                "target": target, # B x T
                }

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward_encoder(
            source,
            padding_mask=padding_mask,
            mask=mask,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["encoder_out"]
        return feature, res["encoder_padding_mask"]

    def get_logits(self, net_output):
        logits_list = net_output["logit_m_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output):
        targets_list = net_output["target_list"]
        targets_list = [x.type(torch.LongTensor) for x in targets_list]
        return targets_list


@register_model("hubert_info_align_phn_mask", dataclass=HubertInfoAlignConfig)
class HubertInfoAlignPhnMaskModel(HubertInfoAlignModel):
    def __init__(
        self,
        hubert_encoder: HubertModel, 
        unit_decoder: TransformerUnitDecoder, 
        cfg: HubertInfoAlignConfig,
        task_cfg: HubertInfoAlignPhnMaskPretrainingConfig,
        tgt_dict: Dictionary,
    ) -> None:
        logger.info(f"HubertInfoAlignPhnMaskModel Config: {cfg}")

        super().__init__(hubert_encoder, unit_decoder, cfg, task_cfg, tgt_dict)

    @classmethod
    def build_model(cls, cfg: HubertInfoAlignConfig, task: HubertInfoAlignPhnMaskPretrainingTask):
        
        assert len(task.dictionaries) == 1
        tgt_dict = task.dictionaries[0]

        # Hubert encoder 
        encoder = cls.build_encoder(cfg.pretrained_hubert_ckpt, cfg, task.cfg, tgt_dict)

        # Autoregressive unit decoder 
        cfg.encoder_embed_dim = cfg.decoder_encoder_embed_dim
        cfg.dropout = cfg.decoder_dropout
        cfg.activation_dropout = cfg.decoder_activation_dropout
        decoder = cls.build_decoder(cfg, tgt_dict)

        # HubertInfoAlign model 
        base_model = HubertInfoAlignPhnMaskModel(encoder, 
                                          decoder, 
                                          cfg, 
                                          task.cfg, 
                                          tgt_dict)

        return base_model

    def apply_encoder_mask_and_hide(self, 
                   x, 
                   padding_mask, 
                   alignment, 
                   mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
                   hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
                  ):
        """compute and apply [MASK] and [HIDE] spans to x. 
        The spans can be sampled at random or pre-specified, and in 
        either way, no two spans overlap. 
        Additional, [MASK] and [HIDE] are based on alignment.

        Exmaples of pre-specifying [MASK] and [HIDE] spans:
            mask_span_selected = [(2, 10), (67, 98)]
            hide_span_selected = [(21, 45)]
        """
        #logger.info(alignment)
        #logger.info(x.shape)
        B, T, C = x.shape
        assert len(mask_span_selected) == 1 and len(hide_span_selected) == 1
        mask_indices, hide_indices = None, None
        if (self.mask_random or self.hide_random or 
            mask_span_selected != [(0, 0)] or 
            hide_span_selected != [(0, 0)]
            ):
            mask_indices, hide_indices = compute_neighboring_phn_mask_and_phn_hide_indices(
                (B, T),
                alignment, 
                padding_mask,
                self.mask_lengths,
                self.hide_lengths,
                mask_random=self.mask_random, 
                hide_random=self.hide_random, 
                mask_span_selected=mask_span_selected, 
                hide_span_selected=hide_span_selected, 
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                mask_dropout=0.0, 
                hide_dropout=0.0, 
            )
            #logger.info(mask_indices[0])
            #logger.info(hide_indices[0])
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            hide_indices = torch.from_numpy(hide_indices).to(x.device)
            x[mask_indices] = self.mask_emb
            x[hide_indices] = self.hide_emb

            return x, mask_indices, hide_indices
        else: 
            return x, None, None 

    def forward(
        self,
        source: torch.Tensor, 
        prev_output_tokens: torch.Tensor, 
        alignment_list: List, 
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_layer: Optional[int] = None,
        mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
        hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
    ):
        assert target_list is None or len(target_list) == 1
        assert len(alignment_list) == 1
        target = target_list[0]
        alignment = alignment_list[0]

        # forward encoder 
        encoder_out = self.forward_encoder(
            source, 
            target, 
            alignment, 
            padding_mask, 
            mask, 
            output_layer, 
            mask_span_selected, 
            hide_span_selected,
        )

        # determine [MASK] regions
        masked_indices = torch.logical_and(~encoder_out["encoder_padding_mask"][0], 
                                            encoder_out["mask_indices"][0])
        bsz = prev_output_tokens.shape[0]
        # determine max length of [MASK] region 
        # this saves space compared to 
        # max_masked_len = self.mask_lengths[-1] * max(self.num_mask_spans)
        max_masked_len = 0
        for i in range(bsz): 
            true_indices = torch.where(masked_indices[i])[0]
            max_masked_len = max(max_masked_len, len(true_indices))
        max_masked_len += max(self.num_mask_spans)-1 # account for the addition of separator token 
        masked_prev_output_tokens = torch.full((bsz, max_masked_len), self.tgt_dict.pad()).to(prev_output_tokens)
        masked_target = torch.full((bsz, max_masked_len), -1).to(prev_output_tokens)

        def extract_masked_consecutive_id(masked_bool, unit_seq): 
            # get indices of all True values
            true_indices = torch.where(masked_bool)[0]
        
            # check if indices form consecutive sequence -- there is only 1 [MASK] region
            if len(true_indices) > 0 and true_indices[-1] - true_indices[0] == len(true_indices) - 1:
                masked_unit_seq = unit_seq[true_indices]
                return masked_unit_seq

            # Below is for cases where there are multiple [MASK] regions. 
            # We want to extract the [MASK] regions separately such that we can insert 
            # a separator token (<eos>) in between them. 

            # initialize variables for sequence tracking
            true_indices_sequences = []
            current_sequence = []
            last_index = None

            # iterate over indices and identify consecutive sequences
            for index in true_indices.tolist():
                if last_index is None:
                    # start new sequence
                    current_sequence.append(index)
                elif index == last_index + 1:
                    # continue current sequence
                    current_sequence.append(index)
                else:
                    # end current sequence and start new sequence
                    true_indices_sequences.append(current_sequence)
                    current_sequence = [index]
                last_index = index

            # append final sequence
            true_indices_sequences.append(current_sequence)

            # filter out sequences of length 1
            true_indices_sequences = [seq for seq in true_indices_sequences if len(seq) > 1]
            true_indices_sequences = [torch.tensor(seq, device=true_indices.device) for seq in true_indices_sequences]
            sequences = [unit_seq[true_idx] for true_idx in true_indices_sequences]

            # insert the separator token in between the [MASK] regions 
            separator = torch.tensor([self.tgt_dict.eos()]).to(true_indices)
            sequences = torch.cat([torch.cat([seq, separator]) for seq in sequences], dim=0)
            return sequences[:-1]

        for i in range(bsz): 
            masked_bool = masked_indices[i]
            #logger.info(masked_bool)
            masked_unit_seq = extract_masked_consecutive_id(masked_bool, prev_output_tokens[i])
            masked_target_seq = extract_masked_consecutive_id(masked_bool, encoder_out["target"][i])

            # deduplicate prediction targets. be careful on the first and last elements.
            masked_unit_seq = torch.cat((masked_unit_seq[:1], torch.unique_consecutive(masked_unit_seq[1:])))
            masked_target_seq = torch.cat((torch.unique_consecutive(masked_target_seq[:-1]), masked_target_seq[-1:]))

            assert len(masked_unit_seq) == len(masked_target_seq)
            masked_prev_output_tokens[i][: len(masked_unit_seq)] = masked_unit_seq
            masked_target[i][: len(masked_target_seq)] = masked_target_seq

        # forward decoder 
        # IMPORTANT: only feed the decoder the [MASK] unit seq instead of the full unit seq!
        decoder_out, _ = self.decoder(
            masked_prev_output_tokens,
            encoder_out=encoder_out,
        )
        #logger.info(decoder_out.shape) # torch.Size([4, 50, 1004])
        #logger.info(masked_prev_output_tokens.shape) # torch.Size([4, 50])
        #logger.info(masked_target.shape) # torch.Size([4, 50])
        #logger.info(masked_target)

        # compute logprob on [MASK] regions
        logit_m = decoder_out[masked_prev_output_tokens != self.tgt_dict.pad()]
        #logger.info(logit_m.shape) # torch.Size([133, 1004])

        target_list_m = masked_target[masked_target != -1]
        #logger.info(target_list_m.shape) # torch.Size([133])
        #logger.info(target_list_m)

        result = {
            "logit_m_list": [logit_m],
            "target_list": [target_list_m], 
        }
        return result

    #def forward_targets(
    #    self,
    #    features: torch.Tensor,
    #    target_list: List[torch.Tensor],
    #    alignment_list: List, 
    #) -> Tuple[torch.Tensor, torch.Tensor, List]:
    #    # Trim features to ensure labels exist and then get aligned labels
    #    feat_tsz = features.size(2)
    #    targ_tsz = min([t.size(1) for t in target_list])
    #    if self.feat2tar_ratio * feat_tsz > targ_tsz:
    #        feat_tsz = int(targ_tsz / self.feat2tar_ratio)
    #        features = features[..., :feat_tsz]
    #    target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
    #    target_list = [t[:, target_inds.long()] for t in target_list]
    #    logger.info(target_inds)
    #    logger.info(target_list[0])
    #    exit()
    #    return features, target_list

    def forward_encoder(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        alignment: Optional[List] = [], 
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_layer: Optional[int] = None,
        mask_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
        hide_span_selected: Optional[List[Tuple[int, int]]] = [(0, 0)], 
    ) -> Dict[str, torch.Tensor]:
        #logger.info(source.shape) # torch.Size([4, 67200])
        #logger.info(target.shape) # torch.Size([4, 209])
        #logger.info(padding_mask.shape)
        features = self.forward_encoder_features(source)
        #logger.info(features.shape) # torch.Size([4, 512, 209])
        if target is not None:
            features, target = self.forward_targets(features, target)

        #logger.info(features.shape) # torch.Size([4, 512, 209])
        #logger.info(target.shape) # torch.Size([4, 209])
        #logger.info(padding_mask)
        #logger.info(padding_mask.shape) # torch.Size([4, 67200])
        #logger.info(mask) # True 
        #logger.info(output_layer) # None

        features = features.transpose(1, 2) # torch.Size([4, 209, 512])
        features = self.encoder.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_encoder_padding_mask(features, padding_mask)

        #logger.info(padding_mask.shape) # torch.Size([4, 209])
        #logger.info(padding_mask)

        if self.encoder.post_extract_proj is not None:
            features = self.encoder.post_extract_proj(features)
        #logger.info(features.shape) # torch.Size([4, 209, 768])

        features = self.encoder.dropout_input(features)


        if mask:
            x, mask_indices, hide_indices = self.apply_encoder_mask_and_hide(features, padding_mask, alignment, mask_span_selected, hide_span_selected)
        else:
            x = features
            mask_indices = None
            hide_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        
        #logger.info(x.shape) # torch.Size([4, 209, 768])
        proj_x = self.encoder.final_proj(x) # D: 768 --> 256
        proj_x = proj_x.transpose(0, 1)
        features = features.transpose(0, 1)

        #logger.info(proj_x.shape) # torch.Size([209, 4, 256])
        #logger.info(features.shape) # torch.Size([209, 4, 768])
        
        return {
                "encoder_out": [proj_x],  # T x B x C
                "encoder_padding_mask": [padding_mask] if padding_mask is not None else [],  # B x T
                "mask_indices": [mask_indices] if mask_indices is not None else [], # B x T
                "hide_indices": [hide_indices] if hide_indices is not None else [], # B x T
                "features": features, # T x B x C
                "target": target, # B x T
                }

    def get_logits(self, net_output):
        logits_list = net_output["logit_m_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output):
        targets_list = net_output["target_list"]
        targets_list = [x.type(torch.LongTensor) for x in targets_list]
        return targets_list


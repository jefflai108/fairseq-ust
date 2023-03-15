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
from fairseq.data.info_align_data_utils import compute_mask_and_hide_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    LAYER_TYPE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.models.hubert.hubert import (
    HubertConfig, 
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_info_align_pretraining import (
    HubertInfoAlignPretrainingConfig,
    HubertInfoAlignPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class HubertInfoAlignConfig(HubertConfig):
    # additional model arguments 
    pretrained_hubert_ckpt: str = field(
        default="", 
        metadata={
            "help": "load a pre-trained Hubert model ckpt for info-align pre-training."
            "If none, info-align training from scratch."
        },
    )

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

@register_model("hubert_info_align", dataclass=HubertInfoAlignConfig)
class HubertInfoAlignModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertInfoAlignConfig,
        task_cfg: HubertInfoAlignPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"HubertInfoAlignModel Config: {cfg}")
        
        self.load_pretrained_hubert_ckpt(cfg.pretrained_hubert_ckpt)

        self.model.train().cuda()

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        #self.feature_extractor = ConvFeatureExtractionModel(
        #    conv_layers=feature_enc_layers,
        #    dropout=0.0,
        #    mode=cfg.extractor_mode,
        #    conv_bias=cfg.conv_bias,
        #)
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        num_class = dictionaries[0].__len__()
        self.pred_proj = nn.Linear(final_dim, num_class)
        nn.init.uniform_(self.pred_proj.weight, a=-0.0, b=1.0)
        nn.init.constant_(self.pred_proj.bias, 0.0)

        # learnable [MASK] and [HIDE] masks
        self.mask_emb = self.model.mask_emb
        self.hide_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        self.hide_emb = self.hide_emb

        # set feature extractor to non-finetunable
        for param in self.model.feature_extractor.parameters():
            param.requires_grad_(False)

        # override the pretrained Hubert's masking parameters 
        self.mask_lengths = cfg.mask_lengths
        self.hide_lengths = cfg.hide_lengths
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_random = cfg.mask_random
        self.hide_random = cfg.hide_random
        self.num_mask_spans = cfg.num_mask_spans
        self.num_hide_spans = cfg.num_hide_spans

    def load_pretrained_hubert_ckpt(self, ckpt_str): 
        """load a pretrained Hubert seed model for info-align pretraining"""
        import fairseq.checkpoint_utils, sys 
        if ckpt_str == "":
            raise Exception("pretrained Hubert ckpt not provided")
        else:
            (
                model,
                cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_str])
            self.model = model[0]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertInfoAlignConfig, task: HubertInfoAlignPretrainingTask):
        """Build a new model instance."""

        model = HubertInfoAlignModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask_and_hide(self, 
                   x, 
                   padding_mask, 
                   target_list, 
                   mask_span_selected=(0, 0), 
                   hide_span_selected=(0, 0), 
                  ):
        """compute and apply [MASK] and [HIDE] spans to x. 
        The spans can be sampled at random or pre-specified, and in 
        either way, no two spans overlap. 

        Exmaples of pre-specifying [MASK] and [HIDE] spans:
            mask_span_selected = (67, 98) 
            hide_span_selected = (1, 45)
        """
        
        B, T, C = x.shape
        mask_indices, hide_indices = None, None
        if (len(self.num_mask_spans) > 0 or 
            len(self.hide_mask_spans) > 0 or 
            mask_span_selected != (0, 0) or 
            hide_span_selected != (0, 0)
            ):
            mask_indices, hide_indices = compute_mask_and_hide_indices(
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
            #print(mask_indices[0])
            #print(hide_indices[0])
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            hide_indices = torch.from_numpy(hide_indices).to(x.device)
            x[mask_indices] = self.mask_emb
            x[hide_indices] = self.hide_emb

            return x, mask_indices, hide_indices
        else: 
            return x, None, None 

    #def compute_nce(self, x, pos, negs):
    #    neg_is_pos = (pos == negs).all(-1)
    #    pos = pos.unsqueeze(0)
    #    targets = torch.cat([pos, negs], dim=0)

    #    logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
    #    logits /= self.logit_temp
    #    if neg_is_pos.any():
    #        logits[1:][neg_is_pos] = float("-inf")
    #    logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    #    return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): 
            features = self.model.feature_extractor(source)
            
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        features, target_list = self.model.forward_targets(features, target_list)

        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        padding_mask = self.model.forward_padding_mask(features, padding_mask)

        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""

        assert target_list is None or len(target_list) == 1

        #print(source.shape) # torch.Size([4, 67200])
        #print(target_list[0].shape) # torch.Size([4, 209])
        features = self.forward_features(source)
        #print(features.shape) # torch.Size([4, 512, 209])
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        #print(features.shape) # torch.Size([4, 512, 209])
        #print(target_list[0].shape) # torch.Size([4, 209])
        #print(padding_mask)
        #print(padding_mask.shape) # torch.Size([4, 67200])
        #print(mask) # True 
        #print(output_layer) # None

        features = features.transpose(1, 2) # torch.Size([4, 209, 512])
        features = self.model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        #print(padding_mask.shape) # torch.Size([4, 209])
        #print(padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)
        #print(features.shape) # torch.Size([4, 209, 768])

        features = self.model.dropout_input(features)

        if mask:
            x, mask_indices, hide_indices = self.apply_mask_and_hide(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None
            hide_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.model.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        
        if features_only:
            return {"x": x, 
                    "padding_mask": padding_mask, 
                    "features": features}
        
        #def compute_pred(proj_x, target, label_embs):
        #    # compute logits for the i-th label set
        #    y = torch.index_select(label_embs, 0, target.long())
        #    negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
        #    if self.target_glu:
        #        y = self.target_glu(y)
        #        negs = self.target_glu(negs)
        #    return self.compute_nce(proj_x, y, negs)

        # predict [MASK] spans 
        masked_indices = torch.logical_and(~padding_mask, mask_indices)
        #print(x.shape) # torch.Size([4, 209, 768])
        #print(masked_indices.shape) # torch.Size([4, 209])
        #print(x[masked_indices].shape) # torch.Size([60, 768])
        proj_x_m = self.model.final_proj(x[masked_indices])
        logit_m = self.pred_proj(proj_x_m)

        #print(masked_indices[0])
        #print(logit_m.shape) # torch.Size([60, 1004])
        #print(proj_x_m.shape) # torch.Size([60, 256])

        #print(target_list[0].shape) # torch.Size([4, 209])
        masked_target_list = target_list[0][masked_indices]
        #print(masked_target_list.shape) # torch.Size([60])
        
        result = {
            "logit_m_list": [logit_m],
            "target_list": [masked_target_list], 
            "padding_mask": padding_mask,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output):
        logits_list = net_output["logit_m_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output):
        targets_list = net_output["target_list"]
        targets_list = [x.type(torch.LongTensor) for x in targets_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        return extra_losses, names


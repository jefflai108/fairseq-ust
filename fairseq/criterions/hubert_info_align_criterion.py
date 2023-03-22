# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional
import copy

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class HubertInfoAlignCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("hubert_info_align", dataclass=HubertInfoAlignCriterionConfig)
class HubertInfoAlignCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.log_keys = [] if log_keys is None else log_keys

        self.eos_idx = task.dictionaries[0].eos()
        self.pad_idx = task.dictionaries[0].pad()

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # construct prev_output_tokens from targets for teacher-forcing
        prev_output_tokens = torch.full_like(sample["target_list"][0], self.pad_idx)
        prev_output_tokens[:, 0].fill_(self.eos_idx)
        prev_output_tokens[:, 1:] = copy.deepcopy(sample["target_list"][0][:, :-1])

        # model forward
        net_output = model(target_list=sample["target_list"], prev_output_tokens=prev_output_tokens, **sample["net_input"])
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output)
        targ_m_list = model.get_targets(net_output)

        assert self.pred_masked_weight == 1.0 and len(logp_m_list) > 0

        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            targ_m = targ_m.to(logp_m.device)
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        with torch.no_grad():
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                targ_m = targ_m.to(logp_m.device)
                pred_class_m = torch.argmax(logp_m, dim=1)
                corr_m = (pred_class_m == targ_m).sum().item()
                count_m = len(targ_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

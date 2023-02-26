# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional
import math

import torch
from torch import nn

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m


class UTSTransformerEncoder(FairseqEncoder):
    def __init__(self, args, src_dict, embed_speaker):
        super().__init__(src_dict)
        self.padding_idx = src_dict.pad()
        self.embed_speaker = embed_speaker
        self.spk_emb_proj = None
        if embed_speaker:
            self.spk_emb_proj = nn.Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.embed_tokens = nn.Embedding(
            len(src_dict), args.encoder_embed_dim, padding_idx=self.padding_idx
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))

        self.transformer_layers = nn.ModuleList(
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            self.ctc_proj = nn.Linear(args.encoder_embed_dim, args.tgt_dict_size)
        
        self.apply(encoder_init)

    def forward(self, src_tokens, src_lengths=None, speaker=None, return_all_hiddens=False, **kwargs):
        x = self.embed_tokens(src_tokens)
        x = x.contiguous()  # B x T x C
        x = self.embed_scale * x # added, ref speech_to_text.s2t_transformer.S2TTransformerEncoder

        padding_mask = src_tokens.eq(self.padding_idx)
        encoder_padding_mask = lengths_to_padding_mask(src_lengths)
        assert (padding_mask == encoder_padding_mask).all()
        
        positions = self.embed_positions(padding_mask)
        x += self.pos_emb_alpha * positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        for layer in self.transformer_layers:
            x, _ = layer(x, padding_mask)
            if not isinstance(x, torch.Tensor):  # extra fields in layer output
                x = x[0]
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.embed_speaker:
            seq_len, bsz, _ = x.size()
            emb = self.embed_speaker(speaker).transpose(0, 1)
            emb = emb.expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask]
            if padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

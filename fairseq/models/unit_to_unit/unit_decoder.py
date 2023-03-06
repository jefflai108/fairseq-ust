# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.hub_interface import S2SHubInterface
from fairseq.models.speech_to_speech.modules import CTCDecoder, StackedEmbedding
from fairseq.models.speech_to_text import S2TTransformerEncoder
from fairseq.models.text_to_speech import TTSTransformerDecoder
from fairseq.models.transformer import (
    Linear, 
    TransformerDecoder, 
    TransformerModelBase,
    RawTransformerDecoder, 
    RawTransformerDecoderBase
)

logger = logging.getLogger(__name__)

class TransformerUnitDecoder(TransformerDecoder):
    """Based on Transformer decoder, with support to decoding stacked units"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]

class RawTransformerUnitDecoder(RawTransformerDecoder):
    """Same as TransformerUnitDecoder, but from RawTransformerDecoder instead of TransformerDecoder"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

class TransformerUnitDecoderCopyMechanism(RawTransformerUnitDecoder):
    """Based on TransformerUnitDecoder, with copy mechanism on an external init lexicon"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )

        # Fairseq dictionary 
        self.tgt_dict = dictionary
        assert self.tgt_dict.bos() == 0
        assert self.tgt_dict.pad() == 1
        assert self.tgt_dict.eos() == 2
        assert self.tgt_dict.unk() == 3
        self.tgt_dict_offset = 4

        # lexical translation arguments
        self.is_copy = args.is_copy
        self.lex_alignment_prob = None
        if self.is_copy and args.lex_alignment_npy:
            self.lex_alignment_prob = np.load(args.lex_alignment_npy)
            self.tgt_vocab_size = self.tgt_dict.__len__()
            assert self.lex_alignment_prob.shape[1] == self.tgt_vocab_size

        # learnable attention weights 
        self.alpha_w = Linear(args.decoder_attention_heads, 1, bias=False)

        # (learnable) gate probability w_{gate}. Init to 0 for training stability.` 
        self.copy_gate = Linear(self.output_embed_dim, 1, bias=True)
        torch.nn.init.zeros_(self.copy_gate.weight)
        torch.nn.init.zeros_(self.copy_gate.bias)

        self.EPS = 1e-7

    def _get_attention_projection(self, src_tokens: Tensor, lexicon: np.ndarray) -> Tensor:
        bsz, src_seq_len = src_tokens.size() 

        # proj is a very sparse matrix in B, T', V size
        proj = np.zeros((bsz, src_seq_len, self.tgt_vocab_size), dtype=np.int64)

        # lexicon is np.ndarray that maps an input token index to output probabilities
        # its shape is (V', V) where V' is the size of the input vocabulary, 
        inputs = src_tokens.cpu().numpy()
        for i in range(bsz):
            proj[i, range(src_seq_len), :] = lexicon[inputs[i, :], :]
        return torch.tensor(proj, dtype=torch.float32, device=src_tokens.device)

    def forward(
        self,
        src_tokens, 
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        attention_token_projection: Optional[Tensor] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        # last decoder layer's encoder_attn:
        #   1. Raw attn weights (pre-softmax).
        #   2. Separately returned for each head. 
        #   3. of shape torch.Size([num_head, batch_size, tgt_seq_len, src_seq_len])
        attn_head_raw_weights = extra["raw_attn"][-1]

        if not features_only:
            bsz, seq_len, d = x.size()
            assert self.n_frames_per_step == 1

            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)

            if self.is_copy and self.lex_alignment_prob is not None:
                attention_token_projection = self._get_attention_projection(src_tokens, self.lex_alignment_prob)

                hidden_states = x
                prediction_logits = self.output_layer(hidden_states.view(bsz, seq_len, d))
                prediction_probs = utils.softmax(prediction_logits, dim=-1, onnx_trace=self.onnx_trace)
               
                # p_gate learned from h_i: \sigmoid(w_gate*hi)
                gate_logits = self.copy_gate(hidden_states)
                gate_probs = F.sigmoid(gate_logits)
                
                # compute \alpha_i^j by weighted sum over attn_head_raw_weights. Be careful with the masking 
                attn_masking = (attn_head_raw_weights == -torch.inf)[0]
                attn_head_raw_weights = torch.where(attn_head_raw_weights == -torch.inf, 0, attn_head_raw_weights)
                attn_raw_weights = self.alpha_w(attn_head_raw_weights.transpose(0, -1)).transpose(0, -1) # (1, batch_size, tgt_seq_len, src_seq_len)
                attn_raw_weights = torch.where(attn_masking, -torch.inf, attn_raw_weights)
                alpha_ij = utils.softmax(attn_raw_weights, dim=-1, onnx_trace=self.onnx_trace).squeeze(0)

                copy_probs = torch.bmm(alpha_ij, attention_token_projection)
                
                # "P(total) = P(gate) * P(copy) + (1-P(gate)) * P(pred)" 
                total_probs = copy_probs * (gate_probs + self.EPS) + prediction_probs * (1 - gate_probs + self.EPS)

                # back to log_space 
                x = torch.log(total_probs)
            else:
                # default, w/o copy mechanism 
                x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
                x = x.view(bsz, seq_len * self.n_frames_per_step, -1)

                if (
                    incremental_state is None and self.n_frames_per_step > 1
                ):  # teacher-forcing mode in training
                    x = x[
                        :, : -(self.n_frames_per_step - 1), :
                    ]  # remove extra frames after <eos>

        return x, extra

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

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
from fairseq.models.unit_to_unit.unit_decoder import TransformerUnitDecoderCopyMechanism
from fairseq.models.unit_to_unit.uts_transformer import UTSTransformerEncoder
from fairseq.models.transformer import Linear, TransformerDecoder, TransformerModelBase

logger = logging.getLogger(__name__)

class TokenLexU2UTransformerMultitaskModelBase(FairseqEncoderDecoderModel):
    @classmethod
    def build_encoder(cls, args, src_dict=None):
        encoder = UTSTransformerEncoder(args, src_dict, args.target_speaker_embed)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_multitask_decoder(cls, args, tgt_dict, in_dim):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            base_multitask_text_transformer_decoder_arch(decoder_args)
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                dictionary=tgt_dict,
                in_dim=in_dim,
            )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    @classmethod
    def build_model(cls, args, task):
        encoder = (
            cls.build_encoder(args, task.source_dictionary)
            if task.args.source_is_code 
            else cls.build_encoder(args)
        )
        decoder = (
            cls.build_decoder(args, task.target_dictionary)
            if task.args.target_is_code
            else cls.build_decoder(args)
        )
        base_model = cls(encoder, decoder)

        # set up multitask decoders
        base_model.multitask_decoders = {}
        for task_name, task_obj in task.multitask_tasks.items():
            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )
            task_decoder = cls.build_multitask_decoder(
                task_obj.args, task_obj.target_dictionary, in_dim
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        return base_model

    def forward_encoder(self, src_tokens, src_lengths, speaker=None, **kwargs):
        return self.encoder(
            src_tokens, src_lengths=src_lengths, tgt_speaker=speaker, **kwargs
        )


@register_model("token_lex_u2ut_transformer")
class TokenLexU2UTTransformerModel(TokenLexU2UTransformerMultitaskModelBase):
    """
    Based on "Direct speech-to-speech translation model https://arxiv.org/abs/2107.05604"
    
    Unit-to-Unit Transformer encoder + Transformer discrete unit decoder. 

    Reference:
        fairseq.models.speech_to_speech.s2s_transformer.S2UTTransformerModel 
    """

    @staticmethod
    def add_args(parser):
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # speaker
        parser.add_argument(
            "--speaker-embed-dim",
            type=int,
            metavar="N",
            help="speaker embedding dimension",
        )

    @classmethod
    def hub_models(cls):
        base_url = "https://dl.fbaipublicfiles.com/speech_matrix/s2s_models"
        source_lang_list = [
            "cs",
            "de",
            "en",
            "es",
            "et",
            "fi",
            "fr",
            "hr",
            "hu",
            "it",
            "nl",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
        ]
        tgt_lang_list = ["en", "es", "fr"]
        import itertools

        pairs = [
            (a, b)
            for a, b in itertools.product(source_lang_list, tgt_lang_list)
            if a != b
        ]
        return {
            f"textless_{src}_{tgt}": f"{base_url}/checkpoint_textless_{src}_{tgt}.tar"
            for src, tgt in pairs
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        task="speech_to_speech",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            task=task,
            **kwargs,
        )
        return S2SHubInterface(x["args"], x["task"], x["models"][0])

    @classmethod
    def build_decoder(cls, args, tgt_dict):
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )

        return TransformerUnitDecoderCopyMechanism(
            args,
            tgt_dict,
            embed_tokens,
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
        )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        return decoder_out

def base_multitask_text_transformer_decoder_arch(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_layers = getattr(args, "decoder_layers", 2)

    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)

    # decoder layer
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)

    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


def base_token_lex_u2ut_transformer_encoder_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)


@register_model_architecture(
    model_name="token_lex_u2ut_transformer", arch_name="token_lex_u2ut_transformer"
)
def u2ut_architecture_base(args):
    base_token_lex_u2ut_transformer_encoder_architecture(args)

    # decoder
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("token_lex_u2ut_transformer", "token_lex_u2ut_transformer_fisher")
def u2ut_architecture_fisher(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    
    u2ut_architecture_base(args)

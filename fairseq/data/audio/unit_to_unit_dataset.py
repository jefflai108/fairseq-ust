# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from fairseq.data import ConcatDataset, Dictionary
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2SDataConfig
from fairseq.data.audio.unit_to_text_dataset import (
    UnitToTextDataset,
    UnitToTextDatasetCreator,
    TextTargetMultitaskData,
    _collate_frames,
    get_features_or_waveform,
)

logger = logging.getLogger(__name__)


@dataclass
class UnitToUnitDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None


class UnitToUnitDataset(UnitToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2SDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: List[str],
        tgt_n_frames: List[int],
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        source_is_code: bool = True,
        target_is_code: bool = True,
        src_dict: Dictionary = None,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
    ):
        src_texts = src_audio_paths if source_is_code else None
        tgt_texts = tgt_audio_paths if target_is_code else None

        super().__init__(
            split,
            is_train_split,
            data_cfg,
            src_audio_paths,
            src_n_frames,
            ids=ids,
            src_dict=src_dict, 
            tgt_dict=tgt_dict,
            src_texts=src_texts, 
            tgt_texts=tgt_texts,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            n_frames_per_step=n_frames_per_step,
        )

        self.src_audio_paths = src_audio_paths
        self.tgt_audio_paths = tgt_audio_paths
        self.src_lens = [s for s in src_n_frames]
        self.tgt_lens = [t // self.n_frames_per_step for t in tgt_n_frames]

        assert not source_is_code or src_dict is not None
        assert not target_is_code or tgt_dict is not None
        self.source_is_code = source_is_code
        self.target_is_code = target_is_code

        assert len(src_audio_paths) == self.n_samples
        assert len(tgt_audio_paths) == self.n_samples
        assert len(src_n_frames) == self.n_samples
        assert len(tgt_n_frames) == self.n_samples

        self.tgt_speakers = None
        if self.cfg.target_speaker_embed:
            samples = UnitToTextDatasetCreator._load_samples_from_tsv(
                self.cfg.target_speaker_embed, split
            )
            spk_emb_dict = {s["id"]: s["speaker_embed"] for s in samples}
            self.tgt_speakers = [spk_emb_dict[id] for id in self.ids]
            assert len(self.tgt_speakers) == self.n_samples

        logger.info(self.__repr__())

    def pack_tgt_units(self, input: torch.Tensor) -> torch.Tensor:
        if self.n_frames_per_step <= 1:
            return input

        offset = 4
        vocab_size = (
            len(self.tgt_dict) - offset
        )  # remove offset from <bos>, <pad>, <eos>, <unk>, which is specific to fairseq dictionary

        assert input.dim() == 1
        stacked_input = (
            input[:-1].view(-1, self.n_frames_per_step) - offset
        )  # remove <eos>
        scale = [
            pow(vocab_size, self.n_frames_per_step - 1 - i)
            for i in range(self.n_frames_per_step)
        ]
        scale = torch.LongTensor(scale).squeeze(0)
        res = input.new((len(input) - 1) // self.n_frames_per_step + 1).fill_(input[-1])
        res[:-1] = (stacked_input * scale).sum(dim=1) + offset

        return res

    def pack_src_units(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def __getitem__(self, index: int) -> UnitToUnitDatasetItem:

        if not self.source_is_code: 
            source = self._get_source_audio(index)
        else: 
            source = self.src_dict.encode_line(
                self.src_audio_paths[index], 
                add_if_not_exist=False,
                append_eos=True,
            ).long()

        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_as_bos:
            # prepend_tgt_lang_tag_as_bos: put tgt_lang_tag as bos of target
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)
        
        if not self.target_is_code:
            target = get_features_or_waveform(self.tgt_audio_paths[index])
            target = torch.from_numpy(target).float()
            target = self.pack_frames(target)
        else:
            target = self.tgt_dict.encode_line(
                self.tgt_audio_paths[index],
                add_if_not_exist=False,
                append_eos=True,
            ).long()
            if self.n_frames_per_step > 1:
                n_tgt_frame = target.size(0) - 1  # exclude <eos>
                keep_n_tgt_frame = n_tgt_frame - n_tgt_frame % self.n_frames_per_step
                target = torch.cat(
                    (
                        target[:keep_n_tgt_frame],
                        target.new_full((1,), self.tgt_dict.eos()),
                    ),
                    dim=0,
                )

        if self.tgt_speakers:
            tgt_spk = get_features_or_waveform(self.tgt_speakers[index])
            tgt_spk = torch.from_numpy(tgt_spk).float()
        else:
            tgt_spk = torch.FloatTensor([])

        return UnitToUnitDatasetItem(
            index=index,
            source=source,
            target=target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
        )

    def _collate_target(self, samples: List[UnitToUnitDatasetItem]) -> torch.Tensor:
        if self.target_is_code:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            # convert stacked units to a single id
            pack_targets = [self.pack_tgt_units(x.target) for x in samples]
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                pack_targets,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            target_lengths = torch.tensor(
                [x.size(0) for x in pack_targets], dtype=torch.long
            )
        else:
            target = _collate_frames([x.target for x in samples], is_audio_input=False)
            bsz, _, d = target.size()
            prev_output_tokens = torch.cat(
                (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            )
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            )

        return target, prev_output_tokens, target_lengths

    def _collate_source(self, samples: List[UnitToUnitDatasetItem]) -> torch.Tensor:
        if self.source_is_code:
            source = fairseq_data_utils.collate_tokens(
                [x.source for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            # convert stacked units to a single id
            pack_sources = [self.pack_src_units(x.source) for x in samples]
            source_lengths = torch.tensor(
                [x.size(0) for x in pack_sources], dtype=torch.long
            )
        else:
            print('not supported')
            sys.exit(1)
            #target = _collate_frames([x.target for x in samples], is_audio_input=False)
            #bsz, _, d = target.size()
            #prev_output_tokens = torch.cat(
            #    (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            #)
            #target_lengths = torch.tensor(
            #    [x.target.size(0) for x in samples], dtype=torch.long
            #)

        return source, source_lengths

    def collater(
        self, samples: List[UnitToUnitDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        #frames = _collate_frames([x.source for x in samples], True)
        #n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)

        # sort samples by descending number of frames
        frames, n_frames = self._collate_source(samples)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, prev_output_tokens, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.target.size(0) for x in samples)

        tgt_speakers = None
        if self.cfg.target_speaker_embed:
            tgt_speakers = _collate_frames(
                [x.target_speaker for x in samples], is_audio_input=True
            ).index_select(0, order)
        
        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": tgt_speakers,  # TODO: unify "speaker" and "tgt_speaker"
        }
        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": tgt_speakers,  # to support Tacotron2 loss for speech-to-spectrogram model
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class UnitToUnitMultitaskDataset(UnitToUnitDataset):
    def __init__(self, *argv):
        super().__init__(*argv)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[UnitToUnitDatasetItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[UnitToUnitDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out


class UnitToUnitDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        source_is_code: bool = True, 
        target_is_code: bool = True,
        source_dictionary: Dictionary = None, 
        target_dictionary: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> UnitToUnitDataset:

        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [
            s[cls.KEY_SRC_AUDIO] 
            if source_is_code 
            else (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() 
            for s in samples
        ]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = len(multitask) > 0
        dataset_cls = (
            UnitToUnitMultitaskDataset if has_multitask else UnitToUnitDataset
        )

        ds = dataset_cls(
            split_name,
            is_train_split,
            data_cfg,
            src_audio_paths,
            src_n_frames,
            tgt_audio_paths,
            tgt_n_frames,
            src_langs,
            tgt_langs,
            ids,
            source_is_code,
            target_is_code,
            source_dictionary,
            target_dictionary,
            n_frames_per_step,
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2SDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        source_is_code: bool = True, 
        target_is_code: bool = True,
        source_dictionary: Dictionary = None, 
        target_dictionary: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> UnitToUnitDataset:
        datasets = []
        for split in splits.split(","):
            samples = UnitToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split,
                is_train_split,
                samples,
                data_cfg,
                source_is_code, 
                target_is_code,
                source_dictionary, 
                target_dictionary,
                n_frames_per_step,
                multitask,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
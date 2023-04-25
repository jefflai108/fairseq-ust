# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import warnings
from typing import Optional, Tuple, List
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

def compute_mask_and_hide_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_length_range: List[int],
    hide_length_range: List[int],
    mask_type: str = "static",
    mask_random: bool = True, 
    hide_random: bool = True, 
    mask_span_selected: List[Tuple[int, int]] = (0, 0), 
    hide_span_selected: List[Tuple[int, int]] = (0, 0),
    num_mask_spans: List[int] = [1], 
    num_hide_spans: List[int] = [1], 
    no_overlap: bool = False,
    min_space: int = 0,
    mask_dropout: float = 0.0,
    hide_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random MASK and HIDE spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        mask_random: mask # of masks specify by num_mask_span at random 
        num_mask_span: number of masked spans
        mask_span_selected: ignore num_mask_span, mask a span specify by the (start span id, end span id)
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    hide = np.full((bsz, all_sz), False)
   
    assert no_overlap is True # all of [MASK] and [HIDE] spans hav no overlaps 
    mask_idcs = []
    hide_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
        else:
            sz = all_sz

        # sample num. of [MASK] and # of [HIDE]
        if len(num_mask_spans) > 0: 
            num_mask = random.choice(num_mask_spans)
        else: 
            num_mask = 0
        if len(num_hide_spans) > 0: 
            num_hide = random.choice(num_hide_spans)
        else: 
            num_hide = 0

        # sample length of [MASK] and length of [HIDE]
        if len(mask_length_range) > 0: 
            mask_length = random.randint(mask_length_range[0], mask_length_range[1])
        if len(hide_length_range) > 0:
            hide_length = random.randint(hide_length_range[0], hide_length_range[1])

        #print(num_mask, num_hide)
        #print(mask_length, hide_length)

        if mask_type == "static":
            mask_lengths = np.full(num_mask, mask_length)
            hide_lengths = np.full(num_hide, hide_length)
        else:
            raise Exception("unknown mask selection " + mask_type)

        if len(mask_length_range) > 0 and sum(mask_length_range) == 0:
            mask_lengths[0] = min(mask_length, sz - 1)
        if len(hide_length_range) > 0 and sum(hide_length_range) == 0:
            hide_lengths[0] = min(hide_length, sz - 1)

        # [MASK] and [HIDE] are either both at random or both pre-specified
        # If specified, they should not overlap. 
        assert mask_random == hide_random 
        if mask_span_selected != [(0, 0)]: 
            assert not any(x in mask_span_selected for x in hide_span_selected)

        # [MASK] span selection
        if mask_random and mask_span_selected == [(0, 0)]:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            mask_parts = [(0, sz)]
            min_mask_length = min(mask_lengths)
            for length in sorted(mask_lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in mask_parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(mask_parts), p=probs)
                s, e = mask_parts.pop(c)
                mask_parts.extend(arrange(s, e, length, min_mask_length))
        else: 
            mask_idc = []
            for _mask_span_selected in mask_span_selected: 
                mask_idc.extend([i for i in range(_mask_span_selected[0], _mask_span_selected[1] + 1)])

        # [HIDE] span selection
        if hide_random and hide_span_selected == [(0, 0)]:
            hide_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                hide_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            hide_parts = mask_parts # excluding selected [MASK] spans
            min_hide_length = min(hide_lengths)
            for length in sorted(hide_lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in hide_parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(hide_parts), p=probs)
                s, e = hide_parts.pop(c)
                hide_parts.extend(arrange(s, e, length, min_hide_length))
        else: 
            hide_idc = []
            for _hide_span_selected in hide_span_selected: 
                hide_idc.extend([i for i in range(_hide_span_selected[0], _hide_span_selected[1] + 1)])

        mask_idc = np.asarray(mask_idc)
        hide_idc = np.asarray(hide_idc)

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
        hide_idcs.append(np.unique(hide_idc[hide_idc < sz]))

    for i, mask_idc in enumerate(mask_idcs):
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )
       
        if mask_idc != []: 
            mask[i, mask_idc] = True

    for i, hide_idc in enumerate(hide_idcs):
        if hide_dropout > 0:
            num_holes = np.rint(len(hide_idc) * hide_dropout).astype(int)
            hide_idc = np.random.choice(
                hide_idc, len(hide_idc) - num_holes, replace=False
            )

        if hide_idc != []:
            hide[i, hide_idc] = True

    return mask, hide 


def compute_neighboring_mask_and_hide_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_length_range: List[int],
    hide_length_range: List[int],
    mask_type: str = "static",
    mask_random: bool = True, 
    hide_random: bool = True, 
    mask_span_selected: List[Tuple[int, int]] = (0, 0), 
    hide_span_selected: List[Tuple[int, int]] = (0, 0),
    num_mask_spans: List[int] = [1], 
    num_hide_spans: List[int] = [1], 
    no_overlap: bool = False,
    min_space: int = 0,
    mask_dropout: float = 0.0,
    hide_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random MASK and HIDE spans for a given shape. Enforce MASK and HIDE are neighboring 

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        mask_random: mask # of masks specify by num_mask_span at random 
        num_mask_span: number of masked spans
        mask_span_selected: ignore num_mask_span, mask a span specify by the (start span id, end span id)
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    hide = np.full((bsz, all_sz), False)
   
    assert no_overlap is True # all of [MASK] and [HIDE] spans hav no overlaps 
    mask_idcs = []
    hide_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
        else:
            sz = all_sz

        # sample num. of [MASK] and # of [HIDE]
        if len(num_mask_spans) > 0: 
            num_mask = random.choice(num_mask_spans)
        else: 
            num_mask = 0
        if len(num_hide_spans) > 0: 
            num_hide = random.choice(num_hide_spans)
        else: 
            num_hide = 0

        # sample length of [MASK] and length of [HIDE]
        if len(mask_length_range) > 0: 
            mask_length = random.randint(mask_length_range[0], mask_length_range[1])
        if len(hide_length_range) > 0:
            hide_length = random.randint(hide_length_range[0], hide_length_range[1])

        #print(num_mask, num_hide)
        #print(mask_length, hide_length)

        if mask_type == "static":
            mask_lengths = np.full(num_mask, mask_length)
            hide_lengths = np.full(num_hide, hide_length)
        else:
            raise Exception("unknown mask selection " + mask_type)

        if len(mask_length_range) > 0 and sum(mask_length_range) == 0:
            mask_lengths[0] = min(mask_length, sz - 1)
        if len(hide_length_range) > 0 and sum(hide_length_range) == 0:
            hide_lengths[0] = min(hide_length, sz - 1)

        # [MASK] and [HIDE] are either both at random or both pre-specified
        # If specified, they should not overlap. 
        assert mask_random == hide_random 
        if mask_span_selected != [(0, 0)]: 
            assert not any(x in mask_span_selected for x in hide_span_selected)

        if mask_random: 
            # First sample [MASK], then sample [HIDE] based on LEFT or RIGHT or BOTH
            HIDE_POS = random.choice(['R', 'L', 'B'])
            if HIDE_POS == 'B': 
                mask_lengths[0] = mask_lengths[0] + hide_lengths[0]
            #print(HIDE_POS)

        # [MASK] span selection
        if mask_random and mask_span_selected == [(0, 0)]:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            if HIDE_POS == 'R':
                mask_parts = [(0, sz-hide_lengths[0])]
            elif HIDE_POS == 'L': 
                mask_parts = [(hide_lengths[0], sz)]
            else: 
                mask_parts = [(0, sz)]

            min_mask_length = min(mask_lengths)
            for length in sorted(mask_lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in mask_parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(mask_parts), p=probs)
                s, e = mask_parts.pop(c)
                mask_parts.extend(arrange(s, e, length, min_mask_length))
        else: 
            mask_idc = []
            for _mask_span_selected in mask_span_selected: 
                mask_idc.extend([i for i in range(_mask_span_selected[0], _mask_span_selected[1] + 1)])

        # [HIDE] span selection
        if hide_random and hide_span_selected == [(0, 0)]:
            hide_idc = []

            if HIDE_POS == 'R':
                hide_idc = list(range(mask_idc[-1]+1, mask_idc[-1]+1+hide_lengths[0]))
            elif HIDE_POS == 'L':
                hide_idc = list(range(mask_idc[0]-hide_lengths[0], mask_idc[0]))
            else: 
                hide_idc = []

        else: 
            hide_idc = []
            for _hide_span_selected in hide_span_selected: 
                hide_idc.extend([i for i in range(_hide_span_selected[0], _hide_span_selected[1] + 1)])

        mask_idc = np.asarray(mask_idc)
        hide_idc = np.asarray(hide_idc)
        #print(mask_idc)
        #print(hide_idc)

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
        hide_idcs.append(np.unique(hide_idc[hide_idc < sz]))

    for i, mask_idc in enumerate(mask_idcs):
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )
       
        if mask_idc != []: 
            mask[i, mask_idc] = True

    for i, hide_idc in enumerate(hide_idcs):
        if hide_dropout > 0:
            num_holes = np.rint(len(hide_idc) * hide_dropout).astype(int)
            hide_idc = np.random.choice(
                hide_idc, len(hide_idc) - num_holes, replace=False
            )

        if hide_idc != []:
            hide[i, hide_idc] = True
    
    return mask, hide 


def compute_neighboring_phn_mask_and_phn_hide_indices(
    shape: Tuple[int, int],
    alignment: List, 
    padding_mask: Optional[torch.Tensor],
    mask_length_range: List[int],
    hide_length_range: List[int],
    mask_type: str = "static",
    mask_random: bool = True, 
    hide_random: bool = True, 
    mask_span_selected: List[Tuple[int, int]] = (0, 0), 
    hide_span_selected: List[Tuple[int, int]] = (0, 0),
    no_overlap: bool = False,
    min_space: int = 0,
    mask_dropout: float = 0.0,
    hide_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random MASK and HIDE spans for a given shape. Enforce MASK and HIDE are neighboring 

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        mask_random: mask # of masks specify by num_mask_span at random 
        num_mask_span: number of masked spans
        mask_span_selected: ignore num_mask_span, mask a span specify by the (start span id, end span id)
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    hide = np.full((bsz, all_sz), False)
   
    assert no_overlap is True # all of [MASK] and [HIDE] spans hav no overlaps 
    mask_idcs = []
    hide_idcs = []
    for i in range(bsz):
        phn_segments = len(alignment[i])
            
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
        else:
            sz = all_sz

        # sample length of [MASK] and length of [HIDE]
        if len(mask_length_range) > 0: 
            mask_length = min(random.randint(mask_length_range[0], mask_length_range[1]), phn_segments - 1)
        if len(hide_length_range) > 0:
            hide_length = min(random.randint(hide_length_range[0], hide_length_range[1]), phn_segments - 1 - mask_length)

        #logger.info(phn_segments)
        #logger.info(mask_length)
        #logger.info(hide_length)
        #logger.info(alignment[i])

        if mask_type == "static":
            mask_lengths = np.full(1, mask_length)
            hide_lengths = np.full(1, hide_length)
        else:
            raise Exception("unknown mask selection " + mask_type)

        if len(mask_length_range) > 0 and sum(mask_length_range) == 0:
            mask_lengths[0] = min(mask_length, sz - 1)
        if len(hide_length_range) > 0 and sum(hide_length_range) == 0:
            hide_lengths[0] = min(hide_length, sz - 1)

        # [MASK] and [HIDE] are either both at random or both pre-specified
        # If specified, they should not overlap. 
        assert mask_random == hide_random 
        if mask_span_selected != [(0, 0)]: 
            assert not any(x in mask_span_selected for x in hide_span_selected)

        if mask_random: 
            # First sample [MASK], then sample [HIDE] based on LEFT or RIGHT or BOTH
            HIDE_POS = random.choice(['R', 'L', 'B'])
            if HIDE_POS == 'B': 
                mask_lengths[0] = mask_lengths[0] + hide_lengths[0]
        #logger.info(HIDE_POS)

        # [MASK] span selection
        if mask_random and mask_span_selected == [(0, 0)]:
            mask_idc = []
            if HIDE_POS == 'R':
                mask_start_segment_idx = np.random.randint(0, phn_segments - mask_lengths[0] - hide_lengths[0])
            elif HIDE_POS == 'L':
                mask_start_segment_idx = np.random.randint(hide_lengths[0], phn_segments - mask_lengths[0])
            else:
                mask_start_segment_idx = np.random.randint(0, phn_segments - mask_lengths[0])

            mask_start_segment = alignment[i][mask_start_segment_idx]
            mask_end_segment = alignment[i][mask_start_segment_idx + mask_lengths[0] - 1]
            mask_idc = [mask_start_segment[0], mask_end_segment[-1]]
        else: 
            #mask_start_segment = alignment[i][mask_span_selected[0][0]]
            #mask_end_segment = alignment[i][mask_span_selected[0][1]]
            #mask_idc = [mask_start_segment[0], mask_end_segment[-1]]
            mask_idc = mask_span_selected[0]
        mask_idc = list(range(mask_idc[0], mask_idc[1]+1))

        #logger.info(mask_idc)

        # [HIDE] span selection
        if hide_random and hide_span_selected == [(0, 0)]:
            hide_idc = []
            if HIDE_POS == 'R':
                hide_start_segment_idx = mask_start_segment_idx + mask_lengths[0]
            elif HIDE_POS == 'L':
                hide_start_segment_idx = mask_start_segment_idx - hide_lengths[0]
            else: 
                hide_start_segment_idx = None 

            if hide_start_segment_idx is not None:
                hide_start_segment = alignment[i][hide_start_segment_idx]
                hide_end_segment = alignment[i][hide_start_segment_idx + hide_lengths[0] - 1]
                hide_idc = [hide_start_segment[0], hide_end_segment[-1]]
        else: 
            #hide_start_segment = alignment[i][hide_span_selected[0][0]]
            #hide_end_segment = alignment[i][hide_span_selected[0][1]]
            #hide_idc = [hide_start_segment[0], hide_end_segment[-1]]
            hide_idc = hide_span_selected[0]
        if hide_idc != []:
            hide_idc = list(range(hide_idc[0], hide_idc[1]+1))

        #logger.info(hide_idc)

        mask_idc = np.asarray(mask_idc)
        hide_idc = np.asarray(hide_idc)
        #logger.info(mask_idc)
        #logger.info(hide_idc)

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
        hide_idcs.append(np.unique(hide_idc[hide_idc < sz]))

    for i, mask_idc in enumerate(mask_idcs):
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )
       
        if mask_idc != []: 
            mask[i, mask_idc] = True

    for i, hide_idc in enumerate(hide_idcs):
        if hide_dropout > 0:
            num_holes = np.rint(len(hide_idc) * hide_dropout).astype(int)
            hide_idc = np.random.choice(
                hide_idc, len(hide_idc) - num_holes, replace=False
            )

        if hide_idc != []:
            hide[i, hide_idc] = True
    
    return mask, hide 

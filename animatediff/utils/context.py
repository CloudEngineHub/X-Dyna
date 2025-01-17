# Adapted from https://github.com/s9roll7/animatediff-cli-prompt-travel/tree/main

# Copyright (c) 2023 animatediff-cli Authors.
# Copyright 2024 ByteDance and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Di Chang on 11/16/2024.
#
# Original file was released under Apache License, Version 2.0, with the full license text
# available at https://github.com/s9roll7/animatediff-cli-prompt-travel/blob/main/LICENSE.md
#
# This modified file is released under the same license.

import numpy as np
from typing import Callable, Optional, List


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
    offset=0,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [(e+offset) % num_frames for e in range(j, j + context_size * context_step, context_step)]


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )

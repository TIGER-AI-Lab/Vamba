# coding=utf-8
# Copyright (c) 2024 Zip Zou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''The state class to record metrics.'''

import weakref
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from numpy import typing as npt
from transformers.training_args import TrainingArguments

from .types import Number


class AdditionalState:

    def __init__(self, args: TrainingArguments) -> None:
        self.metrics: Dict[str, List[Union[Number, torch.Tensor,
                                           npt.NDArray]]] = defaultdict(list)
        self.args = weakref.ref(args)

    def add_metrics(self, **metrics: Union[Number, torch.Tensor, npt.NDArray]):
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def get_metrics(
        self,
        step_scale: float = 1.0,
        gather_func: Optional[Callable[
            [Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]] = None,
        round_digits: Optional[int] = None
    ) -> Dict[str, Number]:
        metrics: Dict[str, List[Number]] = defaultdict(list)
        for k, values in self.metrics.items():
            for value in values:
                if isinstance(value, torch.Tensor):
                    if gather_func is not None:
                        value = gather_func(value).mean().item()
                    else:
                        value = value.mean().cpu().item()
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, (int, float)):
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, np.ndarray):
                    val = value.mean().item()
                    val = val / self.args().gradient_accumulation_steps
                else:
                    val = value
                metrics[k].append(val)

        step_metrics = {
            k: sum(v) / (len(v) / self.args().gradient_accumulation_steps)
            for k, v in metrics.items()
        }
        if round_digits is not None:
            step_metrics = {
                k: round(v, round_digits)
                for k, v in step_metrics.items()
            }

        return step_metrics

    def pop_metrics(
        self,
        gather_func: Optional[Callable[
            [Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]] = None,
        round_digits: Optional[int] = None
    ):
        ret = self.get_metrics(gather_func, round_digits)

        self.clear()

        return ret

    def clear(self):
        self.metrics.clear()

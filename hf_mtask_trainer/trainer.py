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
'''The trainer supporting multiple metrics record.'''

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from .mixins import MultiTaskModuleMixin
from .state import AdditionalState

DataCollator = Callable[[List[Any]], Dict[str, Any]]


def _patching_module_base(module: Module, additional_state: AdditionalState):
    if isinstance(module, Module) and hasattr(
        module, 'supports_report_metrics'
    ) and module.supports_report_metrics and MultiTaskModuleMixin not in module.__class__.__bases__:
        module.__class__.__bases__ = module.__class__.__bases__ + (
            MultiTaskModuleMixin,
        )
        module.report_metrics = partial(
            module.report_metrics, additional_state
        )


class HfMultiTaskTrainer(Trainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, Module]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, Any]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], Any]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Optional[Tuple[Optimizer, LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ):
        self.additional_state = AdditionalState(args)
        if model is not None:
            report_patching = partial(
                _patching_module_base, additional_state=self.additional_state
            )
            model.apply(report_patching)
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer,
            model_init, compute_metrics, callbacks, optimizers,
            preprocess_logits_for_metrics
        )

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        if hasattr(self, 'additional_state'):
            additional_logs = self.additional_state.pop_metrics(
                gather_func=self._nested_gather
            )
        else:
            additional_logs = {}

        epoch = logs.pop('epoch', None)
        logs.update(additional_logs)
        logs['epoch'] = epoch

        output = {
            **logs,
            **{
                "step": self.state.global_step
            }
        }
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

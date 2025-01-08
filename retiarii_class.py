import atexit
import logging
import time
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from subprocess import Popen
from threading import Thread
import time
from typing import Any, List, Optional, Union

import colorama
import psutil

import torch
import torch.nn as nn
import nni.runtime.log
from nni.experiment import Experiment
from nni.experiment.config.training_service import TrainingServiceConfig
from nni.experiment import management, launcher, rest
from nni.experiment.config import utils
from nni.experiment.config.base import ConfigBase
from nni.experiment.pipe import Pipe
from nni.tools.nnictl.command_utils import kill_command

""" from ..converter import convert_to_graph
from ..graph import Model, Evaluator
from ..integration import RetiariiAdvisor
from ..mutator import Mutator
from ..nn.pytorch.mutator import process_inline_mutation
from ..strategy import BaseStrategy
from ..oneshot.interface import BaseOneShotTrainer
 """
_logger = logging.getLogger(__name__)


@dataclass
class RetiariiExeConfig(ConfigBase):
    experiment_name: str = ''
    search_space: Any = ''  # TODO: remove
    trial_command: str = 'python3 -m nni.retiarii.trial_entry'
    trial_code_directory: utils.PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: int = 0
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: Optional[utils.PathLike] = None
    # remove configuration of tuner/assessor/advisor
    training_service: TrainingServiceConfig

    def __init__(self, training_service_platform: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert 'training_service' not in kwargs
            self.training_service = utils.training_service_config_factory(platform = training_service_platform)

    def __setattr__(self, key, value):
        fixed_attrs = {'search_space': '',
                       'trial_command': 'python3 -m nni.retiarii.trial_entry'}
        if key in fixed_attrs and fixed_attrs[key] != value:
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (value == Path('.') or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        self.__dict__[key] = value

    def validate(self, initialized_tuner: bool = False) -> None:
        super().validate()


    @property
    def _canonical_rules(self):
        return _canonical_rules

    @property
    def _validation_rules(self):
        return _validation_rules



_canonical_rules = {'trial_code_directory': utils.canonical_path,
    'max_experiment_duration': lambda value: f'{utils.parse_time(value)}s' if value is not None else None,
    'experiment_working_directory': utils.canonical_path}


_validation_rules = {
    'trial_code_directory': lambda value: (Path(value).is_dir(), f'"{value}" does not exist or is not directory'),
    'trial_concurrency': lambda value: value > 0,
    'trial_gpu_number': lambda value: value >= 0,
    'max_experiment_duration': lambda value: utils.parse_time(value) > 0,
    'max_trial_number': lambda value: value > 0,
    'log_level': lambda value: value in ["trace", "debug", "info", "warning", "error", "fatal"],
    'training_service': lambda value: (type(value) is not TrainingServiceConfig, 'cannot be abstract base class')
}
import logging
import random
from typing import *
__package__ = 'tracegnn.models.gtrace'

from tracegnn.models.gtrace.evaluate import evaluate
from tracegnn.models.gtrace.models.level_model import LevelModel
from .config import ExpConfig

import mltk
import dgl
from loguru import logger
import dgl.dataloading
import torch
import torch.backends.cudnn
import os
import numpy as np

from tracegnn.data import *
from tracegnn.utils import *

from .dataset import TestDataset, TrainDataset

from .trainer import trainer


def init_seed(config: ExpConfig):
    # set random seed to encourage reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)
        random.seed(config.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main(exp: mltk.Experiment):
    # config
    config: ExpConfig = exp.config

    # init
    init_seed(config)
    logger.info(f'Device: {config.device}')

    # Load dataset
    logger.info(f'Loading dataset {config.dataset} ({config.test_dataset})...')
    train_dataset = TrainDataset(config, valid=False)
    val_dataset = TrainDataset(config, valid=True)

    # Check if the test path exists
    test_flag = os.path.exists(os.path.join(r'D:/GraduationProject/GTrace', config.dataset_root_dir, config.dataset, 'processed', config.test_dataset))
    if test_flag:
        test_dataset = TestDataset(config)
    else:
        test_dataset = None

    train_loader = dgl.dataloading.GraphDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    # logger.info('Loading train dataset!')
    val_loader = dgl.dataloading.GraphDataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=True)
    # logger.info('Loading val dataset!')

    if test_dataset is not None:
        test_loader = dgl.dataloading.GraphDataLoader(
            test_dataset, batch_size=config.batch_size)
        # logger.info('Loading test dataset!')
    else:
        test_loader = None
        # logger.info('Loading no test dataset!')

    # # Train
    trainer(config=exp.config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader)

    # Test model simply
    # model = LevelModel(config).to(config.device)
    # load_model = torch.load(r'D:/GraduationProject/GTrace/tracegnn/models/gtrace/model.pth')
    # model.load_state_dict(load_model)
    # print(model)



if __name__ == '__main__':
    with mltk.Experiment(ExpConfig) as exp:
        main(exp)

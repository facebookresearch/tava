# Copyright (c) Meta Platforms, Inc. and affiliates.
"""An abstract engine for training / inference."""
import logging
import os
from abc import abstractmethod

from omegaconf import DictConfig
from tava.utils.training import resume_from_ckpt

LOGGER = logging.getLogger(__name__)


class AbstractEngine:
    def __init__(
        self,
        local_rank: int,
        world_size: int,
        cfg: DictConfig,
    ) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self.cfg = cfg
        self.device = "cuda:%d" % local_rank

        self.init_step = 0
        self.max_steps = self.cfg.max_steps

        # setup dirs
        self.save_dir = os.getcwd()
        self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        LOGGER.info("All files are saved to %s" % self.save_dir)

        # setup model
        self.model, self.optimizer = self.build_model()
        if self.cfg.resume:
            if self.cfg.resume_dir is not None:
                # load other pretrain models.
                _ = resume_from_ckpt(
                    path=self.cfg.resume_dir,
                    model=self.model,
                    step=self.cfg.resume_step,
                    strict=False,
                )
            else:
                # resume training from itself.
                self.init_step = resume_from_ckpt(
                    path=self.ckpt_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.cfg.resume_step,
                    strict=True,
                )
        self.model.to(self.device)

        # setup dataset
        self.dataset, self.meta_data = self.build_dataset()
        LOGGER.info("Initialization done in Rank %d" % self.local_rank)

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self):
        raise NotImplementedError

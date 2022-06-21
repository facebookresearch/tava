# Copyright (c) Meta Platforms, Inc. and affiliates.
import functools
import logging
import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from tava.engines.abstract import AbstractEngine
from tava.utils.evaluation import eval_epoch
from tava.utils.structures import namedtuple_map
from tava.utils.training import (
    clean_up_ckpt,
    compute_psnr_from_mse,
    learning_rate_decay,
    save_ckpt,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


def default_collate_fn(data):
    return data[0]


class Trainer(AbstractEngine):
    def __init__(
        self,
        local_rank: int,
        world_size: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__(local_rank, world_size, cfg)
        # setup tensorboard
        # must be after model resuming for `self.init_step`` to be updated.
        self.tb_writer = SummaryWriter(
            log_dir=self.save_dir, purge_step=self.init_step
        )
        self.tb_writer.add_text("cfg", str(self.cfg), 0)

        self.learning_rate_fn = functools.partial(
            learning_rate_decay,
            lr_init=cfg.lr_init,
            lr_final=cfg.lr_final,
            max_steps=cfg.max_steps,
            lr_delay_steps=cfg.lr_delay_steps,
            lr_delay_mult=cfg.lr_delay_mult,
        )

        if self.cfg.distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
            torch.distributed.barrier(device_ids=[self.local_rank])  # sync

    def build_model(self):
        LOGGER.info("* Creating Model.")
        current_seed = torch.initial_seed()
        torch.manual_seed(1234)  # same parameters for multi-gpus
        model = instantiate(self.cfg.model).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.lr_init,
            weight_decay=self.cfg.weight_decay_mult,
        )
        torch.manual_seed(current_seed)
        return model, optimizer

    def build_dataset(self):
        LOGGER.info("* Creating Dataset.")
        dataset = {
            split: instantiate(self.cfg.dataset, split=split)
            for split in [self.cfg.train_split]
        }
        dataset.update(
            {
                split: instantiate(
                    self.cfg.dataset,
                    split=split,
                    num_rays=None,
                    cache_n_repeat=None,
                )
                for split in self.cfg.eval_splits
            }
        )
        return dataset, None

    def run(self) -> float:  # noqa
        LOGGER.info("Start Running in Rank %d!" % self.local_rank)
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset[self.cfg.train_split],
            num_workers=4,
            batch_size=1,
            collate_fn=default_collate_fn,
        )
        train_dataloader_iter = iter(train_dataloader)

        stats_trace = deque([], maxlen=100)
        reset_timer = True
        is_main_thread = self.local_rank % self.world_size == 0

        for step in range(self.init_step, self.max_steps + 1):
            try:
                data = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                data = next(train_dataloader_iter)
            if reset_timer:
                t_loop_start = time.time()
                reset_timer = False
            lr = self.learning_rate_fn(step)
            stats = self.train_step(data, lr)
            stats_trace.append(stats)

            if is_main_thread and step % self.cfg.print_every == 0:
                avg_loss = sum([s["loss"] for s in stats_trace]) / len(
                    stats_trace
                )
                avg_psnr = sum([s["psnr"] for s in stats_trace]) / len(
                    stats_trace
                )
                steps_per_sec = self.cfg.print_every / (
                    time.time() - t_loop_start
                )
                rays_per_sec = (
                    self.cfg.dataset.num_rays * steps_per_sec * self.world_size
                )
                precision = int(np.ceil(np.log10(self.cfg.max_steps))) + 1
                LOGGER.info(
                    ("{:" + "{:d}".format(precision) + "d}").format(step)
                    + f"/{self.cfg.max_steps:d}: "
                    + "".join(
                        [
                            f"{k}={v:0.4f} (Avg {sum([s[k] for s in stats_trace]) / len(stats_trace):0.4f}), "
                            for k, v in stats.items()
                            if "loss" in k and v > 0
                        ]
                    )
                    + f"lr={lr:0.2e}, "
                    + f"{rays_per_sec:0.0f} rays/sec"
                )
                reset_timer = True
                for k, v in stats.items():
                    self.tb_writer.add_scalar(k, v, step)
                self.tb_writer.add_scalar("train_avg_loss", avg_loss, step)
                self.tb_writer.add_scalar("train_avg_psnr", avg_psnr, step)
                self.tb_writer.add_scalar("learning_rate", lr, step)
                self.tb_writer.add_scalar(
                    "train_steps_per_sec", steps_per_sec, step
                )
                self.tb_writer.add_scalar(
                    "train_rays_per_sec", rays_per_sec, step
                )

            if is_main_thread and step % self.cfg.save_every == 0:
                LOGGER.info("* Saving")
                save_ckpt(self.ckpt_dir, step, self.model, self.optimizer)
                clean_up_ckpt(self.ckpt_dir, 5)

            # evaluation on epoch.
            if (
                self.cfg.eval_every > 0
                and step % self.cfg.eval_every == 0
                and step > 0
            ):
                for eval_split in self.cfg.eval_splits:
                    LOGGER.info("* Evaluation on split %s." % eval_split)
                    val_dataset = self.dataset[eval_split]
                    eval_render_every = math.ceil(
                        len(val_dataset)
                        / (self.world_size * self.cfg.eval_per_gpu)
                    )
                    metrics = eval_epoch(
                        self.model,
                        val_dataset,
                        data_preprocess_func=lambda x: self._preprocess(
                            x, eval_split
                        ),
                        render_every=eval_render_every,
                        test_chunk=self.cfg.test_chunk,
                        save_dir=os.path.join(
                            self.save_dir, self.cfg.eval_cache_dir, eval_split
                        ),
                        local_rank=self.local_rank,
                        world_size=self.world_size,
                    )
                    self.model.train()
                    self.tb_writer.add_scalar(
                        "%s_psnr_eval" % eval_split, metrics["psnr"], step
                    )
                    self.tb_writer.add_scalar(
                        "%s_ssim_eval" % eval_split, metrics["ssim"], step
                    )
                    if is_main_thread:
                        # save the metrics and print
                        with open(
                            os.path.join(
                                self.save_dir, "%s_metrics_otf.txt" % eval_split
                            ),
                            mode="a",
                        ) as fp:
                            fp.write(
                                "step=%d, test_render_every=%d, psnr=%.4f, ssim=%.4f\n"
                                % (
                                    step,
                                    eval_render_every,
                                    metrics["psnr"],
                                    metrics["ssim"],
                                )
                            )
                        LOGGER.info(
                            f"Eval Epoch {step}: "
                            + f"split = {eval_split} "
                            + f"psnr = {metrics['psnr']:.4f} "
                            + f"ssim = {metrics['ssim']:.4f} "
                        )
        LOGGER.info("Finished Training in Rank %d!" % self.local_rank)
        return 1.0

    def _preprocess(self, data, split=None):
        # to gpu
        for k, v in data.items():
            if k == "rays":
                data[k] = namedtuple_map(lambda x: x.to(self.device), v)
            elif isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                pass
        return data

    def train_step(self, data, lr):
        self.model.train()
        data = self._preprocess(data)
        rays = data.pop("rays")
        pixels = data.pop("pixels")

        # Forward
        ret, _ = self.model(rays=rays, randomized=True, **data)
        if len(ret) not in (1, 2):
            raise ValueError(
                "ret should contain either 1 set of output (coarse only), or 2 sets"
                "of output (coarse as ret[0] and fine as ret[1])."
            )

        # The main prediction is always at the end of the ret list.
        rgb, _, _, _ = ret[-1]
        loss = F.mse_loss(rgb, pixels)
        psnr = compute_psnr_from_mse(loss)
        if len(ret) > 1:
            # If there are both coarse and fine predictions, we compute the loss for
            # the coarse prediction (ret[0]) as well.
            rgb_c, _, _, _ = ret[0]
            loss_c = F.mse_loss(rgb_c, pixels)
            psnr_c = compute_psnr_from_mse(loss_c)
        else:
            loss_c = 0.0
            psnr_c = 0.0
        for i, param in enumerate(self.optimizer.param_groups):
            param["lr"] = lr
        self.optimizer.zero_grad()
        loss_all = (
            loss
            + loss_c * self.cfg.loss_coarse_mult
        )
        loss_all.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "psnr": psnr,
            "loss_c": loss_c,
            "psnr_c": psnr_c,
        }

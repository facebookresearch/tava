# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import math
import os

import torch
from hydra.utils import instantiate
from tava.engines.abstract import AbstractEngine
from tava.utils.evaluation import eval_epoch
from tava.utils.structures import namedtuple_map

LOGGER = logging.getLogger(__name__)


class Evaluator(AbstractEngine):

    def build_model(self):
        LOGGER.info("* Creating Model.")
        model = instantiate(self.cfg.model).to(self.device)
        model.eval()
        return model, None

    def build_dataset(self):
        LOGGER.info("* Creating Dataset.")
        dataset = {
            split: instantiate(
                self.cfg.dataset,
                split=split,
                num_rays=None,
                cache_n_repeat=None,
            )
            for split in self.cfg.eval_splits
        }
        meta_data = {
            split: dataset[split].build_pose_meta_info()
            for split in dataset.keys()
        }
        return dataset, meta_data
    
    def _preprocess(self, data, split):
        # to gpu
        for k, v in data.items():
            if k == "rays":
                data[k] = namedtuple_map(lambda x: x.to(self.device), v)
            elif isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                pass
        # update pose info for this frame
        meta_data = self.meta_data[split]
        idx = meta_data["meta_ids"].index(data["meta_id"])
        data["bones_rest"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_rest"]
        )
        data["bones_posed"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_posed"][idx]
        )
        if "pose_latent" in meta_data:
            data["pose_latent"] = meta_data["pose_latent"][idx].to(self.device)
        return data

    def run(self) -> float:  # noqa
        if self.init_step <= 0 and (not os.path.exists(self.cfg.resume_dir)):
            LOGGER.warning(
                "Ckpt not loaded! Please check save_dir: %s or resume_dir: %s." % (
                    self.save_dir, self.cfg.resume_dir
                )
            )
            return 0.

        is_main_thread = self.local_rank % self.world_size == 0

        for eval_split in self.cfg.eval_splits:
            val_dataset = self.dataset[eval_split]
            eval_render_every = math.ceil(
                len(val_dataset) 
                / float(self.world_size * self.cfg.eval_per_gpu)
            )
            LOGGER.info(
                "* Evaluation on split %s. Total %d, eval every %d" % (
                    eval_split, len(val_dataset), eval_render_every
                )
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

            if is_main_thread and self.cfg.compute_metrics:
                str_log = "".join([
                    "%s = %.6f; " % (key, value) 
                    for key, value in metrics.items()
                ])
                LOGGER.info(
                    "Done average: %s" % str_log
                )
                with open(
                    os.path.join(self.save_dir, "%s_metrics.txt" % eval_split), 
                    mode="a"
                ) as fp:
                    fp.write(
                        "step=%d, test_render_every=%d, %s\n" % 
                        (self.init_step, eval_render_every, str_log)
                    )

            if self.cfg.distributed:
                # blocks all GPUs
                torch.distributed.barrier(device_ids=[self.local_rank])
                LOGGER.info("Sync ... Rank %d" % self.local_rank)
        return 1.0

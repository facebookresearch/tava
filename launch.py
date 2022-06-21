#!/usr/bin/python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Launch jobs."""
import logging
import os
import random
from os import path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join(path.dirname(__file__), "configs")


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _distributed_worker(
    local_rank: int,
    world_size: int,
    cfg: DictConfig,
    hydra_config: DictConfig,
) -> float:
    configure_log(hydra_config.job_logging, hydra_config.verbose)
    LOGGER.info("Distributed worker: %d / %d" % (local_rank + 1, world_size))
    if cfg.percision == "float64":
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)
    if cfg.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=local_rank
        )
    _set_random_seed(1234 + local_rank)
    engine = instantiate(cfg.engine, local_rank, world_size, cfg)
    output = engine.run()
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()
    LOGGER.info("Job Done for worker: %d / %d" % (local_rank + 1, world_size))
    return output


def _run(cfg: DictConfig, hydra_config: DictConfig) -> float:
    assert torch.cuda.is_available(), "CUDA device is required!"
    assert cfg.percision in ["float32", "float64"]
    world_size = torch.cuda.device_count()
    if world_size == 1:
        cfg.distributed = False
    LOGGER.info("cfg is:")
    LOGGER.info(OmegaConf.to_yaml(cfg))
    if cfg.distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        world_size = torch.cuda.device_count()
        process_context = torch.multiprocessing.spawn(
            _distributed_worker,
            args=(
                world_size,
                cfg,
                hydra_config,
            ),
            nprocs=world_size,
            join=False,
        )
        try:
            process_context.join()
        except KeyboardInterrupt:
            # this is important.
            # if we do not explicitly terminate all launched subprocesses,
            # they would continue living even after this main process ends,
            # eventually making the OD machine unusable!
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    LOGGER.info("terminating process " + str(i) + "...")
                    process.terminate()
                process.join()
                LOGGER.info("process " + str(i) + " finished")
        return 1.0
    else:
        return _distributed_worker(
            local_rank=0, world_size=1, cfg=cfg, hydra_config=hydra_config
        )


@hydra.main(config_path=CONF_FP, config_name="mipnerf_dyn")
def cli(cfg: Any) -> float:
    hydra_config = HydraConfig.get()
    return _run(cfg, hydra_config)


if __name__ == "__main__":
    cli()

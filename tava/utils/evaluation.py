# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os

import imageio
import numpy as np
import torch
from tava.utils.structures import namedtuple_map
from tava.utils.training import compute_psnr, compute_ssim
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def eval_epoch(
    model,
    dataset,
    data_preprocess_func,
    render_every: int = 1,
    test_chunk: int = 1024,
    save_dir: str = None,
    local_rank: int = 0,
    world_size: int = 1,
):
    """The multi-gpu evaluation function."""
    device = "cuda:%d" % local_rank
    metrics = {
        "psnr": torch.tensor(0.0, device=device),
        "ssim": torch.tensor(0.0, device=device),
    }

    if world_size > 1:
        # sync across all GPUs
        torch.distributed.barrier(device_ids=[local_rank])

    model.eval()
    model = model.module if hasattr(model, "module") else model

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # split tasks across gpus
    index_list_all = list(range(len(dataset)))[::render_every]
    index_list = index_list_all[local_rank::world_size]

    for i, index in enumerate(index_list):
        LOGGER.info(
            "Processing %d/%d in Rank %d!"
            % (i + 1, len(index_list), local_rank)
        )

        data = dataset[index]
        data = data_preprocess_func(data)
        rays = data.pop("rays")
        pixels = data.pop("pixels")

        # forward
        pred_color, pred_depth, pred_acc, pred_warp = render_image(
            model=model,
            rays=rays,
            randomized=False,
            normalize_disp=False,
            chunk=test_chunk,
            **data,
        )
        pred_warp = pred_warp * (pred_acc > 0.001)

        # psnr & ssim
        psnr = compute_psnr(pred_color, pixels, mask=None)
        ssim = compute_ssim(pred_color, pixels, mask=None)
        metrics["psnr"] += psnr
        metrics["ssim"] += ssim

        # save images
        if save_dir is not None:
            img_to_save = torch.cat([pred_color, pixels], dim=1)
            sid, meta_id, cid = (
                data.get("subject_id", ""),
                data.get("meta_id", ""),
                data.get("camera_id", ""),
            )
            image_path = os.path.join(
                save_dir, f"{index:04d}_{sid}_{meta_id}_{cid}.png"
            )
            imageio.imwrite(
                image_path,
                np.uint8(img_to_save.cpu().numpy() * 255.0),
            )
            imageio.imwrite(
                image_path.replace(".png", "_mask.png"),
                np.uint8(pred_acc.cpu().numpy() * 255.0),
            )
            if pred_warp.shape[-1] == 3:
                imageio.imwrite(
                    image_path.replace(".png", ".exr"),
                    np.float32(pred_warp.cpu().numpy()),
                )
            else:
                np.save(
                    image_path.replace(".png", ".npy"),
                    np.float32(pred_warp.cpu().numpy()),
                )

    if world_size > 1:
        # sync across all GPUs
        torch.distributed.barrier(device_ids=[local_rank])
        for key in metrics.keys():
            torch.distributed.all_reduce(
                metrics[key], op=torch.distributed.ReduceOp.SUM
            )
        torch.distributed.barrier(device_ids=[local_rank])

    for key, value in metrics.items():
        metrics[key] = value / len(index_list_all)
    return metrics


@torch.no_grad()
def render_image(model, rays, chunk=8192, **kwargs):
    """Render all the pixels of an image (in test mode).

    Args:
      model: the model of nerf.
      rays: a `Rays` namedtuple, the rays to be rendered.
      chunk: int, the size of chunks to render sequentially.

    Returns:
      rgb: torch.tensor, rendered color image.
      disp: torch.tensor, rendered disparity image.
      acc: torch.tensor, rendered accumulated weights per pixel.
      warp: torch.tensor, correspondance per pixel.
    """
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = namedtuple_map(
        lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
    )
    results = []
    for i in tqdm(range(0, num_rays, chunk)):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_results = model(rays=chunk_rays, **kwargs)[0][-1]
        results.append(chunk_results[0:4])
    rgb, depth, acc, warp = [torch.cat(r, dim=0) for r in zip(*results)]
    return (
        rgb.view((height, width, -1)),
        depth.view((height, width, -1)),
        acc.view((height, width, -1)),
        warp.view((height, width, -1)),
    )

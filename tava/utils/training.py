import logging
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

LOGGER = logging.getLogger(__name__)


def save_ckpt(path, step, model, optimizer):
    """Save the checkpoint."""
    ckpt_path = os.path.join(path, "step-%09d.ckpt" % step)
    if hasattr(model, "module"):
        model = model.module
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    LOGGER.info("Save checkpoint to: %s" % ckpt_path)


def clean_up_ckpt(path, n_keep=5):
    """Clean up the checkpoints to keep only the last few (also keep the best one)."""
    if os.path.exists(path):
        return
    ckpt_paths = sorted(
        [os.path.join(path, fp) for fp in os.listdir(path) if ".ckpt" in fp]
    )
    if len(ckpt_paths) > n_keep:
        for ckpt_path in ckpt_paths[:-n_keep]:
            LOGGER.warning("Remove checkpoint: %s" % ckpt_path)
            os.remove(ckpt_path)


def resume_from_ckpt(path, model, optimizer=None, step=None, strict=True):
    """Resume the model & optimizer from the latest/specific checkpoint.

    Return:
        the step of the ckpt. return 0 if no ckpt found.
    """
    if not os.path.exists(path):
        return 0
    if step is not None:
        ckpt_paths = [os.path.join(path, "step-%09d.ckpt" % step)]
        assert os.path.exists(ckpt_paths[0])
    else:
        ckpt_paths = sorted(
            [os.path.join(path, fp) for fp in os.listdir(path) if ".ckpt" in fp]
        )
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(
            {
                key.replace("module.", ""): value
                for key, value in ckpt_data["model"].items()
            },
            strict=strict
        )
        if optimizer is not None:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        step = ckpt_data["step"]
        LOGGER.info("Load model from checkpoint: %s" % ckpt_path)
    else:
        step = 0
    return step


def learning_rate_decay(
    step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    mse = ((pred - target) ** 2).mean()
    return compute_psnr_from_mse(mse)


def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y : y + h, x : x + w]
        target = target[y : y + h, x : x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1
        )
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
        )
    return ssim

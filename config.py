# Modified from https://github.com/SHI-Labs/OneFormer/blob/4962ef6a96ffb76a76771bfa3e8b3587f209752b/oneformer/config.py#L44
import os
import logging
from typing import Any
from pathlib import Path
from detectron2.config import CfgNode as CN

logger = logging.getLogger(__name__)


def get_config_value(cfg: CN, key: str) -> Any:
    subkeys = key.split(".")
    subcfg = cfg
    for idx, subkey in enumerate(subkeys):
        if idx < len(subkeys) - 1:
            subcfg = subcfg.get(subkey, {})
        else:
            subcfg = subcfg.get(subkey, None)

    return subcfg


def set_config_value(cfg: CN, key: str, value: Any):
    subkeys = key.split(".")
    subcfg = cfg
    for idx, subkey in enumerate(subkeys):
        if idx < len(subkeys) - 1:
            subcfg = subcfg.get(subkey, {})
        else:
            subcfg[subkey] = value


def add_custom_config(cfg):
    _add_wandb_config(cfg)
    _add_mmdet_config(cfg)


def _add_wandb_config(cfg):
    # Config for wandb.init(), attributes are converted to lowercase args
    #   e.g. cfg.WANDB.NAME = MyName -> wandb.init(name=MyName)
    # Exceptions are:
    #   WANDB.ENABLED: Used in setup_wandb() and build_writers() to disable wandb, then popped
    #   WANDB.INIT_IGNORE_CONFIG: Used in setup_wandb() to skip config in init, then popped
    if not hasattr(cfg, "WANDB"):
        cfg.WANDB = CN()
    cfg.WANDB.ENTITY = os.environ.get("WANDB_ENTITY", None)
    cfg.WANDB.PROJECT = None
    cfg.WANDB.NAME = None  # Updated below
    cfg.WANDB.TAGS = None
    cfg.WANDB.DIR = None  # Updated below
    cfg.WANDB.RESUME = "auto"
    cfg.WANDB.ALLOW_VAL_CHANGE = True
    cfg.WANDB.ENABLED = True

    # By default, don't log config with init, just log/save the config file
    # This avoids adding many d2-config keys to Wandb projects which are otherwise mmdetection based
    cfg.WANDB.INIT_IGNORE_CONFIG = True


def _add_mmdet_config(cfg):
    """Additional config items for matching output with mmdetection"""
    # Logger to control logging interval for wandb (detectron2 uses 20 always)
    cfg.LOGGER = CN()
    cfg.LOGGER.INTERVAL = 50

    # Additional solver fields for exiting early, without any other impact to training process
    # Used by CustomAMPTrainer / CustomSimpleTrainer via EpochTrainerMixin (train_loop.py)
    cfg.SOLVER.EARLY_EXIT_ITER = None
    cfg.SOLVER.EARLY_EXIT_EPOCHS = None

    # Only latest 2 checkpoints by default
    cfg.SOLVER.CHECKPOINT_MAX_KEEP = 2

    # Allow us to pass in epochs for SOLVER.MAX_ITER, TEST.EVAL_PERIOD, SOLVER.CHECKPOINT_PERIOD
    # We'll calculate iter-based values in CustomTrainer.__init__() (train_net_custom.py)
    cfg.SOLVER.MAX_EPOCHS = None  # Converted to SOLVER.MAX_ITER
    cfg.TEST.EVAL_PERIOD_EPOCHS = None  # Converted to TEST.EVAL_PERIOD
    cfg.SOLVER.CHECKPOINT_PERIOD_EPOCHS = None  # Converted to SOLVER.CHECKPOINT_PERIOD

    # For DATALOADER.SAMPLER_TRAIN == RandomSubsetEpochTrainingSampler
    cfg.DATALOADER.TRAIN_RANDOM_SUBSET_RATIO = None
    cfg.DATALOADER.TRAIN_RANDOM_SUBSET_SIZE = None

    # To use RandomSubsetInferenceSampler
    cfg.DATALOADER.TEST_RANDOM_SUBSET_RATIO = None
    cfg.DATALOADER.TEST_RANDOM_SUBSET_SIZE = None

    # For restarting run in debug mode
    cfg.RESTART_RUN = False


def update_config_epochs(cfg: CN, steps_per_epoch: int):
    cfg.defrost()
    _update_iter_from_epochs(
        cfg=cfg,
        epoch_key="SOLVER.MAX_EPOCHS",
        iter_key="SOLVER.MAX_ITER",
        steps_per_epoch=steps_per_epoch,
    )
    _update_iter_from_epochs(
        cfg=cfg,
        epoch_key="TEST.EVAL_PERIOD_EPOCHS",
        iter_key="TEST.EVAL_PERIOD",
        steps_per_epoch=steps_per_epoch,
    )
    _update_iter_from_epochs(
        cfg=cfg,
        epoch_key="SOLVER.CHECKPOINT_PERIOD_EPOCHS",
        iter_key="SOLVER.CHECKPOINT_PERIOD",
        steps_per_epoch=steps_per_epoch,
    )
    cfg.freeze()


def update_custom_config(cfg: CN, world_size: int):
    _update_wandb_config(cfg)
    _update_train_dataloader_config(cfg)
    _update_model_config(cfg=cfg, world_size=world_size)


def _update_train_dataloader_config(cfg):
    # Use our EpochTrainingSampler
    sampler = cfg.DATALOADER.SAMPLER_TRAIN
    train_subset_ratio = cfg.DATALOADER.TRAIN_RANDOM_SUBSET_RATIO
    train_subset_size = cfg.DATALOADER.TRAIN_RANDOM_SUBSET_SIZE
    expected_samplers = [
        None,
        "TrainingSampler",
        "RandomSubsetTrainingSampler",
        "EpochTrainingSampler",
        "RandomSubsetEpochTrainingSampler",
    ]
    if sampler in expected_samplers:
        # Switch to our epoch-based sampler
        if train_subset_ratio is not None or train_subset_size is not None:
            cfg.DATALOADER.SAMPLER_TRAIN = "RandomSubsetEpochTrainingSampler"
        else:
            cfg.DATALOADER.SAMPLER_TRAIN = "EpochTrainingSampler"
    else:
        raise RuntimeError(
            f"Expected cfg.DATALOADER.SAMPLER_TRAIN in {expected_samplers}, found"
            f" cfg.DATALOADER.SAMPLER_TRAIN={cfg.DATALOADER.SAMPLER_TRAIN}"
        )


def _update_wandb_config(cfg):
    if cfg.WANDB.DIR is None:
        cfg.WANDB.DIR = cfg.OUTPUT_DIR
    cfg.WANDB.NAME = Path(cfg.OUTPUT_DIR).name


def _update_model_config(cfg, world_size: int):
    # Convert SyncBN to BN
    def _convert_sync_bn_to_bn(subcfg: CN):
        for key in list(subcfg.keys()):
            if isinstance(subcfg[key], CN):
                _convert_sync_bn_to_bn(subcfg[key])
            elif isinstance(subcfg[key], str) and "NORM" in key and subcfg[key] == "SyncBN":
                subcfg[key] = "BN"

    if world_size == 1:
        _convert_sync_bn_to_bn(cfg)


def _update_iter_from_epochs(cfg: CN, epoch_key: str, iter_key: str, steps_per_epoch: int):
    epoch_val = get_config_value(cfg=cfg, key=epoch_key)
    if epoch_val is not None:
        assert epoch_val > 0, f"Expected cfg.{epoch_key} > 0, found cfg.{epoch_key}={epoch_val}"
        iter_val = get_config_value(cfg=cfg, key=iter_key)
        new_iter_val = int(epoch_val * steps_per_epoch)
        if iter_val not in [0, -1, None]:
            logger.warning(
                f"Found both cfg.{epoch_key}={epoch_val} and cfg.{iter_key}={iter_val}."
                f" Setting cfg.{iter_key} -> {epoch_val} epochs * {steps_per_epoch} steps per epoch"
                f" = {new_iter_val}."
            )
        set_config_value(cfg=cfg, key=iter_key, value=new_iter_val)

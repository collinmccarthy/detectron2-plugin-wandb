# Modified from https://github.com/SHI-Labs/OneFormer/blob/4962ef6a96ffb76a76771bfa3e8b3587f209752b/oneformer/config.py#L44
import os
import logging
import socket
import re
from typing import Tuple
from typing import Any, Optional
from pathlib import Path
from argparse import Namespace

from detectron2.engine import default_argument_parser
from detectron2.config import CfgNode as CN
import detectron2.utils.comm as comm

logger = logging.getLogger(__name__)


def get_config_value(cfg: CN, key: str, default: Optional[Any] = None) -> Any:
    subkeys = key.split(".")
    subcfg = cfg
    for idx, subkey in enumerate(subkeys):
        if idx < len(subkeys) - 1:
            subcfg = subcfg.get(subkey, {})
        else:
            subcfg = subcfg.get(subkey, default)

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
    cfg.WANDB.TAGS = []
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
    cfg.SOLVER.SLURM_REQUEUE_NUM_EPOCHS = None

    # Only latest 2 checkpoints by default
    cfg.SOLVER.CHECKPOINT_MAX_KEEP = 2
    cfg.SOLVER.CHECKPOINT_BEST_METRICS = ["panoptic_seg/PQ", "sem_seg/mIoU", "bbox/AP", "segm/AP"]
    cfg.SOLVER.CHECKPOINT_BEST_METRICS_WANDB_SAVE = ["panoptic_seg/PQ"]

    # Allow us to pass in epochs for SOLVER.MAX_ITER, TEST.EVAL_PERIOD, SOLVER.CHECKPOINT_PERIOD
    # We'll calculate iter-based values in CustomTrainer.__init__() (train_net_custom.py)
    cfg.SOLVER.MAX_EPOCHS = None  # Converted to SOLVER.MAX_ITER
    cfg.TEST.EVAL_PERIOD_EPOCHS = None  # Converted to TEST.EVAL_PERIOD
    cfg.SOLVER.CHECKPOINT_PERIOD_EPOCHS = None  # Converted to SOLVER.CHECKPOINT_PERIOD

    # Support FP16 testing (necessary for Mapillary Vistas)
    cfg.TEST.EVAL_FP16 = False

    # Support passing in a list of epochs/iters for eval, on top of EVAL_PERIOD_EPOCHS
    cfg.TEST.EVAL_EXPLICIT_EPOCHS = []
    cfg.TEST.EVAL_EXPLICIT_ITERS = []

    # For DATALOADER.SAMPLER_TRAIN == RandomSubsetEpochTrainingSampler
    cfg.DATALOADER.TRAIN_RANDOM_SUBSET_RATIO = None
    cfg.DATALOADER.TRAIN_RANDOM_SUBSET_SIZE = None

    # To use RandomSubsetInferenceSampler
    cfg.DATALOADER.TEST_RANDOM_SUBSET_RATIO = None
    cfg.DATALOADER.TEST_RANDOM_SUBSET_SIZE = None

    # Other dataloader defaults not used in d2
    cfg.DATALOADER.PERSISTENT_WORKERS = True  # This with pq_compute_multi_core() can slow us down?
    cfg.DATALOADER.PIN_MEMORY = False  # True can cause issues on our servers occasionally

    # For restarting run in debug mode
    cfg.RESTART_RUN = False

    # (Optional) Add datasets dir being used, so it's clear from looking at config files
    cfg.DETECTRON2_DATASETS = os.environ.get("DETECTRON2_DATASETS", None)


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
        epoch_key="TEST.EVAL_EXPLICIT_EPOCHS",
        iter_key="TEST.EVAL_EXPLICIT_ITERS",
        steps_per_epoch=steps_per_epoch,
    )
    _update_iter_from_epochs(
        cfg=cfg,
        epoch_key="SOLVER.CHECKPOINT_PERIOD_EPOCHS",
        iter_key="SOLVER.CHECKPOINT_PERIOD",
        steps_per_epoch=steps_per_epoch,
    )
    cfg.freeze()


def update_custom_config(args: Namespace, cfg: CN):
    _update_wandb_config(cfg)
    _update_train_dataloader_config(cfg=cfg, auto_workers=args.auto_workers)
    _update_model_config(cfg=cfg, world_size=(args.num_machines * args.num_gpus))


def _get_num_cpus() -> int:
    # Following https://github.com/pytorch/pytorch/blob/6dc54fe8d670a3ff15f6ba49929deb0202e93948/torch/utils/data/dataloader.py#L534
    num_cpus = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except Exception:
            pass

    if num_cpus is None and os.cpu_count() is not None:
        num_cpus = os.cpu_count()  # Same as multiprocessing.cpu_count()

    if num_cpus is None:
        raise ValueError(
            "Failed to get number of CPUs to automatically set number of workers."
            " Manually set --num-workers=<num_cpus_per_gpu>."
        )
    return num_cpus


def _update_train_dataloader_config(cfg: CN, auto_workers: bool):
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

    if auto_workers:  # Use max number of CPUs available per GPU
        gpus_per_node = comm.get_local_size()
        num_cpus = _get_num_cpus()
        cfg.DATALOADER.NUM_WORKERS = num_cpus // gpus_per_node


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
        if isinstance(epoch_val, (tuple, list)):
            if not all([v > 0 for v in epoch_val]):
                assert False, f"Expected cfg.{epoch_key} > 0, found cfg.{epoch_key}={epoch_val}"
            new_iter_val = [int(v * steps_per_epoch) for v in epoch_val]
        else:
            assert epoch_val > 0, f"Expected cfg.{epoch_key} > 0, found cfg.{epoch_key}={epoch_val}"
            new_iter_val = int(epoch_val * steps_per_epoch)

        iter_val = get_config_value(cfg=cfg, key=iter_key)
        if iter_val not in [0, -1, None, list(), tuple()]:
            logger.warning(
                f"Found both cfg.{epoch_key}={epoch_val} and cfg.{iter_key}={iter_val}."
                f" Setting cfg.{iter_key} -> {epoch_val} epochs * {steps_per_epoch} steps per epoch"
                f" = {new_iter_val}."
            )
        set_config_value(cfg=cfg, key=iter_key, value=new_iter_val)


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def parse_dist_url(dist_url: str, num_machines: int) -> Tuple[str, int]:
    # Current dist_url == 'auto' from _launch isn't working every time (port in use) - use increment method instead
    default_port = 29400  # From torch.distributed.launcher.api.elastic_launch
    if dist_url == "auto":
        # Can't use find_free_port() for multi-node runs
        dist_url = "tcp://127.0.0.1"
        dist_port = _find_free_port() if num_machines == 1 else default_port
    else:
        split_url = dist_url.split(":")
        if len(split_url) not in [2, 3]:
            raise RuntimeError(
                f"Expected --dist-url like 'tcp://<url>:<port>' or '<url>:<port>',"
                f" found dist_url={dist_url}"
            )
        port_search = re.search(".*:(\d+)$", dist_url)
        dist_url = ":".join(split_url[:-1])
        dist_port = int(port_search.group(1)) if port_search is not None else default_port

    return dist_url, dist_port


def parse_args():
    parser = default_argument_parser()

    parser.add_argument(
        "--detectron2-datasets",
        "--detectron2_datasets",
        type=str,
        default=None,
        help="Override DETECTRON2_DATASETS environment variable for top-level data directory",
    )
    parser.add_argument(
        "--auto-workers",
        "--auto_workers",
        action="store_true",
        default=False,
        help="Override DATALOADER.NUM_WORKERS with maximum CPUs per GPU on the machine",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="Run testing instead of training",
    )
    parser.add_argument(
        "--nnodes", type=int, help="Dummy arg, used with torchrun, ignored for detectron2"
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        default="Dummy arg, used with torchrun, ignored for detectron2. Use --num-gpus instead.",
    )

    args = parser.parse_args()

    if args.nnodes is not None and args.nnodes != 1:
        raise RuntimeError(
            f"Found --nnodes={args.nnodes}. Expected 1 or None for detectron2. Multi-node DDP must"
            f" use torchrun and manually launch on each node."
        )

    if args.nproc_per_node is not None and args.nproc_per_node != args.num_gpus:
        raise RuntimeError(
            f"Found --nproc-per-node={args.nproc_per_node}. This is ignored in detectron2 and must"
            f" equal --num-gpus={args.num_gpus}."
        )

    # Override data dir
    if args.detectron2_datasets is not None:
        os.environ["DETECTRON2_DATASETS"] = args.detectron2_datasets

    # Override detectron DDP args with torchrun vars if they exist, just for consistency
    nproc_per_node = os.environ.get("LOCAL_WORLD_SIZE", None)
    if nproc_per_node is not None:
        args.num_gpus = int(nproc_per_node)

    nnodes = os.environ.get("GROUP_WORLD_SIZE", None)
    if nnodes is not None:
        args.num_machines = int(nnodes)

    node_rank = os.environ.get("GROUP_RANK", None)
    if node_rank is not None:
        args.machine_rank = int(node_rank)

    dist_addr = os.environ.get("MASTER_ADDR", None)
    dist_port = os.environ.get("MASTER_PORT", None)
    if dist_addr is not None and dist_port is not None:
        args.dist_url = f"tcp://{dist_addr}:{dist_port}"

    return args

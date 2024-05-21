# Extension of https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/utils/events.py
import os
import wandb
import json
import torch
import re
from typing import Dict, Any, Iterable, Union, List, Optional
from collections import defaultdict
from pathlib import Path
from detectron2.utils import comm
from detectron2.utils.events import (
    EventWriter,
    JSONWriter,
    CommonMetricPrinter,
    get_event_storage,
)


def setup_wandb(cfg, args):
    wandb_enabled = cfg.get("WANDB", {}).get("ENABLED", False)
    wandb_restart = cfg.get("WANDB", {}).get("RESTART_RUN", False)
    if comm.is_main_process() and wandb_enabled:
        init_args = {
            k.lower(): v for k, v in cfg.WANDB.items() if isinstance(k, str) and k not in ["config"]
        }
        init_args.pop("enabled")  # Not part of init, used for this fn and Trainer.build_writers()

        output_dir = init_args.get("dir", None)
        if output_dir is None:
            raise RuntimeError(f"Missing wandb output dir. Set WANDB.DIR in config")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # only include most related part to avoid too big table
        # TODO: add configurable params to select which part of `cfg` should be saved in config
        if "init_ignore_config" in init_args:
            # Don't add config, just pop the key (not used by wandb.init(), only here)
            init_args.pop("init_ignore_config")
        elif "config_exclude_keys" in init_args:
            init_args["config"] = cfg
            init_args["config"]["d2_cfg_file"] = args.config_file
        else:
            init_args["config"] = {
                "d2_model": cfg.MODEL,
                "d2_solver": cfg.SOLVER,
                "d2_cfg_file": args.config_file,
            }
        if ("name" not in init_args) or (init_args["name"] is None):
            init_args["name"] = os.path.basename(args.config_file)

        if wandb_restart and "resume" in init_args:
            init_args["resume"] = False

        wandb.init(**init_args)

        # Save config file
        # Have to rename, see issue
        config = Path(output_dir, "config.yaml")  # Default detectron2 name
        if config.exists():
            renamed_config = config.with_name("config.yml")
            renamed_config.unlink(missing_ok=True)
            config.rename(renamed_config)
            wandb.save(str(config), base_path=str(config.parent), policy="now")


class BaseRule(object):
    def __call__(self, target):
        return target


class IsIn(BaseRule):
    def __init__(self, keywords: str):
        if not isinstance(keywords, (tuple, list)):
            keywords = [keywords]
        self.keywords = keywords

    def __call__(self, target):
        return any([keyword in target for keyword in self.keywords])


class StartsWith(BaseRule):
    def __init__(self, keywords: Union[str, Iterable[str]]):
        if not isinstance(keywords, (tuple, list)):
            keywords = [keywords]
        self.keywords = keywords

    def __call__(self, target):
        return any([target.startswith(keyword) for keyword in self.keywords])


class Equals(BaseRule):
    def __init__(self, keywords: Union[str, Iterable[str]]):
        if not isinstance(keywords, (tuple, list)):
            keywords = [keywords]
        self.keywords = keywords

    def __call__(self, target):
        return any([keyword == target for keyword in self.keywords])


class Prefix(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return "/".join([self.keyword, target])


class EventWriterMixin:
    def __init__(
        self, *args, eval_str: str = "val", wandb_skip_keys: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._last_write = -1
        self._group_rules = [
            (StartsWith(["panoptic_seg", "bbox", "segm", "sem_seg"]), Prefix(eval_str)),
            (Equals(["lr"]), Prefix("train")),
            (IsIn(["loss", "_lr"]), Prefix("train")),
            (IsIn("/"), BaseRule()),
        ]
        default_wandb_skip_keys_regex = [  # Skip keys for wandb that we don't use in mmdetection
            "rank_data_time",
            "data_time",
            "time",
            "eta_seconds",
            "epoch_iter",
            r"min\(IoU, B-Iou\)-[a-zA-Z]+",
        ]
        self._wandb_skip_keys_regex = wandb_skip_keys or default_wandb_skip_keys_regex

    def get_new_scalars(self, wandb_writer: bool) -> Dict[int, Dict]:
        storage = get_event_storage()

        def _group_name(scalar_name):  # From OneFormer wandb
            for rule, op in self._group_rules:
                if rule(scalar_name):
                    return op(scalar_name)
            return scalar_name

        # Extension of detectron2 JSONWriter to format keys in same way for Wandb and JSON
        scalars_per_iter: Dict[int, Dict] = {}
        for key, (val, storage_iter) in storage.latest().items():
            if storage_iter <= self._last_write:
                continue

            if storage_iter not in scalars_per_iter:
                scalars_per_iter[storage_iter] = dict(iter=storage_iter)

            key_with_prefix = _group_name(key)
            scalars_per_iter[storage_iter][key_with_prefix] = val

        for storage_iter in list(scalars_per_iter.keys()):
            scalars_per_iter[storage_iter] = self._finalize_scalars_like_mmdetection(
                scalars=scalars_per_iter[storage_iter], wandb_writer=wandb_writer
            )

        if len(scalars_per_iter) > 0:
            all_iters = sorted(scalars_per_iter.keys())
            self._last_write = max(all_iters)

        return scalars_per_iter

    def _finalize_scalars_like_mmdetection(
        self, scalars: Dict[str, Any], wandb_writer: bool
    ) -> dict:

        # Update scalars to log same keys as mmdetection does
        for key in list(scalars.keys()):
            key_split = key.split("/")
            last_key = key_split[-1]
            wandb_skip_key = any(
                [
                    re.match(pattern=pattern, string=last_key) is not None
                    for pattern in self._wandb_skip_keys_regex
                ]
            )

            if wandb_writer and wandb_skip_key:
                scalars.pop(key)

            elif last_key == "total_loss":
                new_last_key = "loss"
                new_key = "/".join(key_split[:-1] + [new_last_key])
                scalars[new_key] = scalars.pop(key)

            elif last_key.startswith("loss"):
                # Change loss keys like 'train/loss_mask_1' to 'train/d1.loss_mask'
                # Following syntax from https://stackoverflow.com/a/8157317/12422298
                new_last_key = re.sub(pattern=r"(loss.+)_(\d+)", repl=r"d\2.\1", string=last_key)
                new_key = "/".join(key_split[:-1] + [new_last_key])
                scalars[new_key] = scalars.pop(key)

            elif "panoptic_seg" in key_split:  # Panoptic eval
                # e.g. 'val/panoptic_seg/PQ' -> 'val/coco/PQ'
                new_key = key.replace("panoptic_seg/", "coco/")
                scalars[new_key] = scalars.pop(key)

            elif any(inst_key in key_split for inst_key in ["bbox", "segm"]):  # Instance eval
                per_class = re.match(pattern=rf"AP-[a-zA-Z]+", string=last_key) is not None
                if wandb_writer and per_class:
                    scalars.pop(key)  # Drop per-class AP, e.g. 'val/bbox/AP-person'
                else:
                    ap_key = "/".join(key_split[:-1]) + "/AP"  # e.g. 'val/bbox/AP' or 'val/segm/AP'
                    if not key.startswith(ap_key):
                        raise RuntimeError(f"Expected key={key} to start with {ap_key}")

                    ap_rem_key = key.replace(ap_key, "")  # e.g. 'val/bbox/AP50' -> '50'
                    if len(ap_rem_key) > 0:
                        ap_rem_key = ap_rem_key.replace("-", "_")  # e.g. '-person' -> '_person'
                        if not ap_rem_key.startswith("_"):
                            ap_rem_key = f"_{ap_rem_key}"

                    new_ap_key = ap_key.replace("/AP", "_mAP")  # eg 'val/bbox/AP' -> 'val/bbox_mAP'
                    new_ap_key = f"{new_ap_key}{ap_rem_key}"  # eg 'val/bbox_mAP_person'

                    # e.g. 'val/bbox/AP' -> 'val/coco/bbox_mAP'
                    # or   'val/bbox/AP50' -> 'val/coco/bbox_mAP_50'
                    # or   'val/bbox/AP-person' -> 'val/coco/bbox_mAP_person'
                    new_key = new_ap_key.replace("bbox", "coco/bbox").replace("segm", "coco/segm")
                    scalars[new_key] = scalars.pop(key)

            elif "sem_seg" in key_split:  # Semantic seg eval
                per_class = re.match(pattern=f"[a-zA-Z]+-[a-zA-Z]+", string=last_key) is not None
                if wandb_writer and per_class:
                    scalars.pop(key)  # Drop per-class IoU, e.g. 'sem_seg/mIoU-person'
                else:
                    # e.g. 'sem_seg/mIoU' -> 'coco/mIoU'
                    new_key = "/".join(key_split[:-1] + [last_key]).replace("sem_seg/", "coco/")
                    scalars[new_key] = scalars.pop(key)

        return scalars


class CustomWandbWriter(EventWriterMixin, EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, enabled: bool = True, *args, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._enabled = enabled
        super().__init__(*args, **kwargs)

    def write(self):
        if not self._enabled:
            return

        storage = get_event_storage()
        scalars_per_iter = self.get_new_scalars(wandb_writer=True)
        for scalars in scalars_per_iter.values():
            # storage.put_{image,histogram} is only meant to be used by
            # tensorboard writer. So we access its internal fields directly from here.
            if len(storage._vis_data) >= 1:
                scalars["image"] = [
                    wandb.Image(img, caption=img_name)
                    for img_name, img, step_num in storage._vis_data
                ]
                # Storage stores all image data and rely on this writer to clear them.
                # As a result it assumes only one writer will use its image data.
                # An alternative design is to let storage store limited recent
                # data (e.g. only the most recent image) that all writers can access.
                # In that case a writer may not see all image data if its period is long.
                storage.clear_images()

            if len(storage._histograms) >= 1:

                def create_bar(tag, bucket_limits, bucket_counts, **kwargs):
                    data = [[label, val] for (label, val) in zip(bucket_limits, bucket_counts)]
                    table = wandb.Table(data=data, columns=["label", "value"])
                    return wandb.plot.bar(table, "label", "value", title=tag)

                scalars["hist"] = [create_bar(**params) for params in storage._histograms]

                storage.clear_histograms()

            if len(scalars) == 0:
                return

            # Don't log with `step=storage_iter` so if we resume from an old step we don't get a
            #   warning about overwriting history; we always add 'iter' key anyway to record step
            wandb.log(scalars, commit=True)

    def close(self):
        if self._enabled:
            wandb.finish()


class CustomJSONWriter(EventWriterMixin, JSONWriter):
    """
    Update keys when logging to match wandb keys, in case we use json output.
    """

    def write(self):
        scalars_per_iter = self.get_new_scalars(wandb_writer=False)

        for scalars in scalars_per_iter.values():
            self._file_handle.write(json.dumps(scalars, sort_keys=True) + "\n")

        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass


class CustomCommonMetricPrinter(CommonMetricPrinter):
    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            avg_data_time = storage.history("data_time").avg(
                storage.count_samples("data_time", self._window_size)
            )
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            avg_data_time = None
            last_data_time = None
        try:
            avg_iter_time = storage.history("time").global_avg()
            last_iter_time = storage.history("time").latest()
        except KeyError:
            avg_iter_time = None
            last_iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # Update: Add epoch / epoch_float
        try:
            epoch = int(storage.history("epoch").latest())
            epoch_iter = int(storage.history("epoch_iter").latest())
            epoch_float = f'{storage.history("epoch_float").latest():.2f}'
        except KeyError:
            epoch = "N/A"
            epoch_iter = "N/A"
            epoch_float = "N/A"

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            str.format(
                " {eta}iter: {iter}  epoch: {epoch}  epoch_iter: {epoch_iter}"
                + "  epoch_float: {epoch_float}  {losses}  {non_losses}  {avg_time}{last_time}"
                + "{avg_data_time}{last_data_time} lr: {lr}  {memory}",
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                epoch=epoch,
                epoch_iter=epoch_iter,
                epoch_float=epoch_float,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(
                            k, v.median(storage.count_samples(k, self._window_size))
                        )
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                non_losses="  ".join(
                    [
                        "{}: {:.4g}".format(
                            k, v.median(storage.count_samples(k, self._window_size))
                        )
                        for k, v in storage.histories().items()
                        if "[metric]" in k
                    ]
                ),
                avg_time=(
                    "time: {:.4f}  ".format(avg_iter_time) if avg_iter_time is not None else ""
                ),
                last_time=(
                    "last_time: {:.4f}  ".format(last_iter_time)
                    if last_iter_time is not None
                    else ""
                ),
                avg_data_time=(
                    "data_time: {:.4f}  ".format(avg_data_time) if avg_data_time is not None else ""
                ),
                last_data_time=(
                    "last_data_time: {:.4f}  ".format(last_data_time)
                    if last_data_time is not None
                    else ""
                ),
                lr=lr,
                memory=("max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else ""),
            )
        )

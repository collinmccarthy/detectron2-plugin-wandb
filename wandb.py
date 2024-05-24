# Extension of https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/utils/events.py
import os
import logging
import shutil
import signal
from typing import List, Optional
from pathlib import Path

import wandb
from torch.utils.data import DataLoader
from detectron2.utils import comm
from detectron2.utils.events import (
    EventWriter,
    get_event_storage,
)

from .events import EventWriterMixin
from .timeout import Timeout

# Some methods use a "quiet" logger (only logs to file, for SIGTERM handling)
std_logger = logging.getLogger(__name__)
quiet_logger = logging.getLogger("signal_safe")  # See d2_plugin.train_net.py, setup_loggers()


class CustomWandbWriter(EventWriterMixin, EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    _initialized = False
    _closed = True

    def write(self):
        if CustomWandbWriter._closed or not CustomWandbWriter._initialized:
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
        self.close_wandb()

    @classmethod
    def setup_wandb(cls, cfg, args):
        # Remove extra flags for setup that aren't part of wandb.init()
        enabled = cfg.get("WANDB", {}).pop("ENABLED", False)
        restart_run = cfg.get("RESTART_RUN", False)
        if comm.is_main_process() and enabled:
            init_args = {
                k.lower(): v
                for k, v in cfg.WANDB.items()
                if isinstance(k, str) and k not in ["config"]
            }

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

            if restart_run:
                # Don't remove old wandb dirs in case we have issues and want to revisit
                init_args["resume"] = False
            elif "resume" not in init_args:
                init_args["resume"] = "auto"

            wandb.init(**init_args)
            cls._initialized = True
            cls._closed = False

            # Save config file
            # Have to rename, see issue https://github.com/wandb/wandb/issues/7654
            config = Path(output_dir, "config.yaml")  # Default detectron2 name
            if config.exists():
                renamed_config = config.with_name("config.yml")
                renamed_config.unlink(missing_ok=True)
                config.rename(renamed_config)
                wandb.save(str(config), base_path=str(config.parent), policy="now")

        comm.synchronize()

    @classmethod
    def close_wandb(
        cls,
        exit_code: int = 0,
        preempting: bool = False,
        quiet: bool = False,
        dataloaders: Optional[List[DataLoader]] = None,
    ):
        if not cls._initialized:
            return

        if comm.is_main_process():
            logger = quiet_logger if quiet else std_logger

            if cls._closed:
                logger.info(
                    f"Wandb backend is already closed, ignoring close_wandb(exit_code={exit_code})"
                )
                return

            # Allow up to 10 minutes for checkpoints to upload
            # This should be plenty, and will prevent any hangs from huge checkpoint files
            timeout_min = 5

            logger.info(
                f"Wandb backend is not closed. Calling wandb.finish(exit_code={exit_code})"
                f" with a {timeout_min} minute timeout."
            )

            if preempting:
                wandb.mark_preempting()

            # Need to shutdown dataloader workers otherwise wandb.finish() will trigger a dataloader
            # error to be raised
            if dataloaders is not None:
                _quietly_shutdown_dataloaders(dataloaders, quiet=quiet)

            # If we exit with code = 0 the 'wandb-resume.json' file is removed
            # We need to keep it around to always enable resuming, even by increasing num epochs
            resume_file = Path(wandb.run.settings.resume_fname)
            resume_file_backup = _backup_resume_file(resume_file, quiet=quiet)

            try:
                timeout_sec = 60 * timeout_min
                timeout_msg = (
                    f"Reached {timeout_min}-min timeout calling"
                    f" wandb.finish(exit_code={exit_code}). Raising SIGKILL."
                )
                with Timeout(
                    timeout_sec=timeout_sec,
                    timeout_msg=timeout_msg,
                    logger=logger,
                    signal_num=signal.SIGKILL,
                ):
                    wandb.finish(exit_code=exit_code, quiet=quiet)
                    logger.info(f"Succesfully closed wandb with exit_code={exit_code}.")
                    cls._closed = True

            finally:
                if not cls._closed:
                    logger.info(f"Failed to close wandb (timed out or an exception was thrown).")

                # Finally will not be reached if we timed out, but should be reached if another
                #   exception is thrown
                if resume_file_backup.exists() and not resume_file.exists():
                    logger.info(
                        f"Renaming backup resume file: {resume_file_backup.name} -> {resume_file.name}"
                    )
                    os.rename(resume_file_backup, resume_file)
                elif resume_file_backup.exists():
                    logger.info(f"Resume file still exists, removing backup resume file")
                    os.remove(resume_file_backup)

        comm.synchronize()


def _backup_resume_file(resume_file: Path, quiet: bool = False) -> Path:
    logger = quiet_logger if quiet else std_logger
    resume_file_backup = resume_file.with_name(
        resume_file.name.replace(resume_file.suffix, f"_backup{resume_file.suffix}")
    )
    if resume_file.exists():
        logger.info(
            f"Backing up current resume file: {resume_file.name} -> {resume_file_backup.name}"
        )
        shutil.copyfile(resume_file, resume_file_backup)

    return resume_file_backup


def _quietly_shutdown_dataloaders(dataloaders: List[DataLoader], quiet: bool = False):
    logger = quiet_logger if quiet else std_logger
    for idx, dataloader in enumerate(dataloaders):
        if hasattr(dataloader, "_iterator") and dataloader._iterator is not None:
            logger.info(f"Quietly shutting down dataloader {idx+1}/{len(dataloaders)}")
            try:
                if hasattr(dataloader._iterator, "_shutdown_workers"):
                    dataloader._iterator._shutdown_workers()
                del dataloader._iterator

            except Exception:
                pass

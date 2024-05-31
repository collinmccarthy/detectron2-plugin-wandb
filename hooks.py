import time
import datetime
import pprint
from typing import Dict, Optional, List
from pathlib import Path
from detectron2.engine.hooks import LRScheduler, BestCheckpointer, HookBase, EvalHook
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage, EventStorage

from .wandb import CustomWandbWriter


class CustomLRScheduler(LRScheduler):
    """Extension of LRScheduler to log learning rates of each model component separately"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_param_groups: Optional[Dict[str, Dict]] = None

    def after_step(self):
        if hasattr(self.trainer, "compute_unique_lr_groups"):
            unique_lr_groups = self.trainer.compute_unique_lr_groups()
            scalars = {f"{key}_lr": val for key, val in unique_lr_groups.items()}
            self.trainer.storage.put_scalars(**scalars, smoothing_hint=False)
        else:
            lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
            self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()


class CustomBestCheckpointer(BestCheckpointer):
    """Extension of BestCheckpointer to save best checkpoint with wandb"""

    def __init__(self, *args, save_wandb: bool = True, **kwargs):
        self._save_wandb = save_wandb
        super().__init__(*args, **kwargs)

    def _best_checking(self):
        if not comm.is_main_process():  # Non-main won't have val results (val metric missing)
            return

        # Throw an exception if value is missing, don't want to accidentally not save best ckpt
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            raise RuntimeError(
                f"Failed to find val_metric={self._val_metric} in latest storage results."
                f"\nAvailable keys: {pprint.pformat(self.trainer.storage.latest().keys())}."
                f"\nMake sure this value is logged before CustomBestCheckpointer hook."
            )

        latest_metric, metric_iter = metric_tuple

        saved: bool = False

        # Don't add "_{metric_iter:07d}" to stem or we'll always add new ones (locally and wandb)
        # We don't currently have a way to track and erase old checkpoints (this is fine for now)
        checkpoint_stem = f"{self._file_prefix}_{self._val_metric.replace('/', '_')}"
        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(checkpoint_stem, **additional_state)
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
                saved = True
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(checkpoint_stem, **additional_state)
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is "
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
            saved = True
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )

        if saved and self._save_wandb:
            checkpoint_path = Path(self._checkpointer.save_dir, f"{checkpoint_stem}.pth")
            CustomWandbWriter.save_checkpoint(checkpoint_path)


class ETAHook(HookBase):
    def __init__(self, max_iter: int, early_exit_iter: Optional[int] = None):
        self._max_iter = max_iter
        self._early_exit_iter = early_exit_iter
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _log_eta(self) -> Optional[str]:
        # From detectron2.CommonMetricPrinter, but adds our 'early_eta_seconds' metric
        storage = get_event_storage()
        iteration = storage.iter

        eta_seconds: Optional[int]
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)

        except KeyError:
            # estimate eta on our own - more noisy
            eta_seconds = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
            self._last_write = (iteration, time.perf_counter())

        if eta_seconds is not None:
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)

            early_exit_iter = (
                self._early_exit_iter if self._early_exit_iter is not None else self._max_iter
            )
            early_eta_seconds = eta_seconds * (early_exit_iter / self._max_iter)
            storage.put_scalar("early_eta_seconds", early_eta_seconds, smoothing_hint=False)

    def after_step(self):
        self._log_eta()  # Only log after step, not after train


class EpochIterHook(HookBase):
    def _log(self, log_iter: int):
        storage = get_event_storage()
        epoch = self.trainer._trainer.epoch(log_iter)
        epoch_iter = self.trainer._trainer.epoch_iter(log_iter)
        epoch_float = self.trainer._trainer.epoch_float(log_iter)
        storage.put_scalars(
            cur_iter=storage.iter,
            iter=log_iter,
            epoch=epoch,
            epoch_iter=epoch_iter,
            epoch_float=epoch_float,
        )

    def after_step(self):
        # Here trainer.iter not yet incremented by 1 (e.g. after 10 iter, trainer.iter=9)
        # Want to log iter as _next_ iter (e.g. after 10 iter, iter=10)
        # Then log epoch/epoch_iter/epoch_float to match this (e.g. after 10 iter, epoch_iter=10)
        self._log(log_iter=self.trainer.iter + 1)

    def after_train(self):
        # Here trainer.iter is already incremented by 1 (e.g. after 100k iter, trainer.iter=100k)
        # Want to log iter as current iter in the same way
        self._log(log_iter=self.trainer.iter)


class CustomEvalHook(EvalHook):
    def __init__(self, *args, eval_iters: Optional[List[int]] = None, **kwargs):
        if eval_iters is None:
            eval_iters = [-1]
        else:
            if not isinstance(eval_iters, (list, tuple)):
                eval_iters = [eval_iters]

        if not all([isinstance(i, int) for i in eval_iters]):
            raise RuntimeError(
                f"Expected eval_iters to be all integers, found eval_iters={eval_iters}"
            )

        self._iters = eval_iters
        super().__init__(*args, **kwargs)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and (next_iter % self._period == 0 or next_iter in self._iters):
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

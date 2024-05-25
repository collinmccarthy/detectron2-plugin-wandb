import pprint
from typing import Dict, Optional
from pathlib import Path
from detectron2.engine.hooks import LRScheduler, BestCheckpointer
import detectron2.utils.comm as comm

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

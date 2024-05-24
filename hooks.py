from typing import Dict, Optional
from detectron2.engine.hooks import LRScheduler


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

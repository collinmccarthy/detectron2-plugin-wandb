import logging
import numpy as np
from typing import Mapping, Optional, Union
import torch
from torch.utils.data import DataLoader
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.data.common import AspectRatioGroupedDataset, MapDataset
from detectron2.data.samplers import TrainingSampler, RandomSubsetTrainingSampler

from .distributed_sampler import EpochTrainingSampler, RandomSubsetEpochTrainingSampler

logger = logging.getLogger(__name__)


def parse_dataloader(
    data_loader: Union[AspectRatioGroupedDataset, MapDataset, DataLoader]
) -> Union[int, float]:
    # Non-trivial to get the correct number of dataset samples, throw exception if unknown class
    # From detectron2.data.build.py, build_batch_data_loader() there are three return types:
    #   - torch.utils.data.DataLoader (when aspect_ratio_grouping=False)
    #   - AspectRatioGroupedDataset (when aspect_ratio_grouping=True and collate_fn is None)
    #   - MapDataset(dataset=AspectRatioGroupedDataset) (when aspect_ratio_grouping=True and collate_fn is not None)
    pytorch_dataloader: DataLoader
    batch_size: int
    if type(data_loader) == AspectRatioGroupedDataset:
        pytorch_dataloader = data_loader.dataset
        batch_size = data_loader.batch_size  # AspectRatioGroupedDataset.batch_size
    elif type(data_loader) == MapDataset:
        grouped_dataset = data_loader.dataset
        if type(grouped_dataset) != AspectRatioGroupedDataset:
            raise RuntimeError(
                f"Expected type(self.data_loader.dataset) == AspectRatioGroupedDataset,"
                f" found {type(data_loader.dataset)}"
            )
        pytorch_dataloader = grouped_dataset.dataset
        batch_size = grouped_dataset.batch_size
    elif type(data_loader) == DataLoader:
        pytorch_dataloader = data_loader
        batch_size = data_loader.batch_size
    else:
        raise NotImplementedError(
            f"Unexpected type(self.data_loader). Expected type in"
            f" ['MapDataset', 'AspectRatioGroupedDataset', 'DataLoader'],"
            f" found {type(data_loader)}"
        )

    if type(pytorch_dataloader) != DataLoader:
        raise RuntimeError(
            f"Expected type(pytorch_dataloader) == DataLoader,"
            f" found {type(pytorch_dataloader)}. Update EpochTrainerMixin to correctly select"
            f" the DataLoader object from within trainer.data_loader"
        )

    sampler = pytorch_dataloader.dataset.sampler
    steps_per_epoch: Union[int, float]
    if type(sampler) == TrainingSampler:
        # Infinite sampler, not on epoch boundaries (so steps_per_epoch is float)
        steps_per_epoch: float = sampler._size / (batch_size * sampler._world_size)
    elif type(sampler) in [EpochTrainingSampler, RandomSubsetEpochTrainingSampler]:
        steps_per_epoch: int = sampler.padded_size / (batch_size * sampler._world_size)
        if not float(steps_per_epoch).is_integer():
            raise RuntimeError(
                f"Expected steps_per_epoch to be integer-valued for type {type(sampler)}, found"
                f" steps_per_epoch={steps_per_epoch}"
            )
        steps_per_epoch = int(steps_per_epoch)
    elif type(sampler) == RandomSubsetTrainingSampler:
        # Infinite sampler, not on epoch boundaries (so steps_per_epoch is float)
        steps_per_epoch: float = sampler._size_subset / (
            batch_size * sampler._world_size
        )
    else:
        raise NotImplementedError(f"Unexpected sampler type: {type(sampler)}.")

    return steps_per_epoch, batch_size


class EpochTrainerMixin:
    def __init__(self, steps_per_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steps_per_epoch = steps_per_epoch

    def epoch(self, iter: Optional[int] = None) -> int:
        iter = self.iter if iter is None else iter
        return int(iter / self._steps_per_epoch)

    def epoch_iter(self, iter: Optional[int] = None) -> int:
        iter = self.iter if iter is None else iter
        return int(iter - (self.epoch(iter) * self._steps_per_epoch))

    def epoch_float(self, iter: Optional[int] = None) -> float:
        return self.epoch(iter) + self.epoch_iter(iter) / self._steps_per_epoch

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:

        iter = self.iter if iter is None else iter

        # Update: Add epoch and epoch_iter based on dataloader and global iter
        epoch = self.epoch(iter)
        epoch_iter = self.epoch_iter(iter)
        epoch_float = self.epoch_float(iter)

        if (iter + 1) % self.gather_metric_period == 0:
            try:
                EpochTrainerMixin.write_metrics(
                    loss_dict, data_time, iter, epoch, epoch_iter, epoch_float, prefix
                )
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        epoch: int,
        epoch_iter: int,
        epoch_float: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

            # Updated: log epoch / epoch_float
            storage.put_scalars(
                cur_iter=cur_iter,
                epoch=epoch,
                epoch_iter=epoch_iter,
                epoch_float=epoch_float,
            )


class CustomAMPTrainer(EpochTrainerMixin, AMPTrainer):
    pass


class CustomSimpleTrainer(EpochTrainerMixin, SimpleTrainer):
    pass

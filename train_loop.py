import logging
import numpy as np
from typing import Mapping, Optional, Union
import torch
from torch.utils.data import DataLoader
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.data.common import AspectRatioGroupedDataset, MapDataset
from detectron2.data.samplers import TrainingSampler, RandomSubsetTrainingSampler, InferenceSampler

from .distributed_sampler import (
    EpochTrainingSampler,
    RandomSubsetEpochTrainingSampler,
    RandomSubsetInferenceSampler,
)

logger = logging.getLogger(__name__)


def get_epoch(iter: int, steps_per_epoch: int) -> int:
    return int(iter / steps_per_epoch)


def get_epoch_iter(iter: int, steps_per_epoch: int) -> int:
    return int(iter - (get_epoch(iter, steps_per_epoch) * steps_per_epoch))


def get_epoch_float(iter: int, steps_per_epoch: int) -> int:
    return get_epoch(iter, steps_per_epoch) + (
        get_epoch_iter(iter, steps_per_epoch) / steps_per_epoch
    )


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

    if hasattr(pytorch_dataloader.dataset, "sampler"):
        sampler = pytorch_dataloader.dataset.sampler
    elif hasattr(pytorch_dataloader.batch_sampler, "sampler"):
        sampler = pytorch_dataloader.batch_sampler.sampler
    else:
        raise RuntimeError(f"Failed to extract sampler from dataloader")

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
    elif type(sampler) in [InferenceSampler, RandomSubsetInferenceSampler]:
        steps_per_epoch: float = len(sampler) / (batch_size * sampler._world_size)
    elif type(sampler) == RandomSubsetTrainingSampler:
        # Infinite sampler, not on epoch boundaries (so steps_per_epoch is float)
        steps_per_epoch: float = sampler._size_subset / (batch_size * sampler._world_size)
    else:
        raise NotImplementedError(f"Unexpected sampler type: {type(sampler)}.")

    return pytorch_dataloader, steps_per_epoch, batch_size


class EpochTrainerMixin:
    def __init__(self, steps_per_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steps_per_epoch = steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    def epoch(self, iter: Optional[int] = None) -> int:
        iter = self.iter if iter is None else iter
        return get_epoch(iter=iter, steps_per_epoch=self._steps_per_epoch)

    def epoch_iter(self, iter: Optional[int] = None) -> int:
        iter = self.iter if iter is None else iter
        return get_epoch_iter(iter=iter, steps_per_epoch=self._steps_per_epoch)

    def epoch_float(self, iter: Optional[int] = None) -> float:
        iter = self.iter if iter is None else iter
        return get_epoch_float(iter=iter, steps_per_epoch=self._steps_per_epoch)


class CustomAMPTrainer(EpochTrainerMixin, AMPTrainer):
    pass


class CustomSimpleTrainer(EpochTrainerMixin, SimpleTrainer):
    pass

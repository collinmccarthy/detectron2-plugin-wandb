import itertools
import logging
import math
from typing import Optional
import torch

from torch.utils.data.sampler import Sampler
from detectron2.data.samplers.distributed_sampler import TrainingSampler
from detectron2.utils import comm

logger = logging.getLogger(__name__)


def _get_size_subset(
    size: int,
    subset_ratio: Optional[float] = None,
    subset_size: Optional[int] = None,
) -> int:
    if (subset_ratio is None and subset_size is None) or (
        subset_ratio is not None and subset_size is not None
    ):
        raise RuntimeError(
            f"Expected exactly one of subset_ratio or subset_size to be set, found"
            f" subset_raito={subset_ratio} and subset_size={subset_size}"
        )

    if subset_ratio is not None and not 0 < subset_ratio <= 1:
        raise RuntimeError(
            f"Expected subset_ratio > 0 and <= 1, found subset_ratio={subset_ratio}"
        )

    if subset_size is not None and (
        subset_size <= 0 or not float(subset_size).is_integer()
    ):
        raise RuntimeError(f"Expected subset_size to be an integer >= 1")

    if subset_size is not None:
        return min(int(subset_size), size)
    else:
        return int(size * subset_ratio)


class EpochTrainingSampler(TrainingSampler):
    """
    Extension of TrainingSampler to pad indices to a multiple of effective batch size, as we do in
    mmdetection. This ensures the training steps per epoch is an integer value, so we can easily
    convert epochs to iterations.
    """

    def __init__(
        self,
        dataset_size: int,
        total_batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ):
        for size, size_str in [
            (dataset_size, "dataset_size"),
            (total_batch_size, "total_batch_size"),
        ]:
            if not isinstance(size, int):
                raise TypeError(
                    f"TrainingSampler({size_str}=) expects an int. Got type {type(size)}."
                )
            if size <= 0:
                raise ValueError(
                    f"TrainingSampler({size_str}=) expects a positive int. Got {size}."
                )

        self._round_up = round_up
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._dataset_size = dataset_size
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._total_batch_size = total_batch_size

        # Modified from mmengine.dataset.sampler.py, DefaultSampler
        # Round up to be a multiple of total_batch_size (aka effective batch size)
        if self._round_up:
            self._padded_size = (
                math.ceil(dataset_size / self._total_batch_size)
                * self._total_batch_size
            )
        else:
            self._padded_size = dataset_size

        logger.info(
            f"EpochTrainingSampler using dataset_size={self._dataset_size},"
            f"  round_up={self._round_up}, effective_batch_size={self._total_batch_size}"
            f" -> padded_size={self._padded_size}"
        )

    @property
    def padded_size(self):
        return self._padded_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _get_indices(self, generator: torch.Generator):
        if self._shuffle:
            indices = torch.randperm(self._dataset_size, generator=generator).tolist()
        else:
            indices = torch.arange(self._dataset_size).tolist()

        if self._round_up:
            indices = (indices * int(self._padded_size / len(indices) + 1))[
                : self._padded_size
            ]
        return indices

    def _infinite_indices(self):
        g = torch.Generator()
        if self._seed is not None:
            g.manual_seed(self._seed)
        while True:
            indices = self._get_indices(generator=g)
            yield from indices


class RandomSubsetEpochTrainingSampler(EpochTrainingSampler):
    def __init__(
        self,
        dataset_size: int,
        total_batch_size: int,
        subset_ratio: Optional[float] = None,
        subset_size: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            subset_ratio (float): the ratio of subset data to sample from the underlying dataset
            shuffle (bool): whether to shuffle the indices or not
            seed_shuffle (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            seed_subset (int): the seed to randomize the subset to be sampled.
                Must be the same across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        super().__init__(
            dataset_size=dataset_size,
            total_batch_size=total_batch_size,
            shuffle=shuffle,
            seed=seed,
            round_up=round_up,
        )
        self._size_subset = _get_size_subset(
            size=dataset_size, subset_ratio=subset_ratio, subset_size=subset_size
        )

        assert (
            self._size_subset > 0
        ), "Subset size should be > 0 if dataset_size > 0 and subset_ratio > 0"

        # Want to always use the same subset of indices, so select them here
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed)
            indices = torch.randperm(self._dataset_size, generator=g)
        else:
            indices = torch.arange(self._dataset_size)

        self._indices_subset = indices[: self._size_subset]

        logger.info(
            f"RandomSubsetEpochTrainingSampler using size_subset={self._size_subset}"
        )

        # Modified from mmengine.dataset.sampler.py, DefaultSampler
        # Round up to be a multiple of total_batch_size (aka effective batch size)
        if self._round_up:
            self._padded_size = (
                math.ceil(self._size_subset / self._total_batch_size)
                * self._total_batch_size
            )
        else:
            self._padded_size = self._size_subset

    def _get_indices(self, generator: torch.Generator):
        if self._shuffle:
            shuffle_indices = torch.randperm(self._size_subset, generator=generator)
            indices = self._indices_subset[shuffle_indices].tolist()
        else:
            indices = self._indices_subset.tolist()

        if self._round_up:
            indices = (indices * int(self._padded_size / len(indices) + 1))[
                : self._padded_size
            ]
        return indices


class RandomSubsetInferenceSampler(Sampler):
    def __init__(
        self,
        size: int,
        subset_ratio: Optional[float] = None,
        subset_size: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        subset_size = _get_size_subset(
            size=size, subset_ratio=subset_ratio, subset_size=subset_size
        )
        self._size = subset_size

        # Want to always use the same subset of indices, so select them here
        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(size, generator=g)
        else:
            indices = torch.arange(size)

        self._indices_subset = indices[:subset_size]
        assert self._size == len(self._indices_subset)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(self._world_size, self._rank)

        logger.info(f"RandomSubsetInferenceSampler using size_subset={subset_size}")

    def _get_local_indices(self, world_size, rank):
        """Same as InferenceSampler but uses self._indices_subset"""
        shard_size = self._size // world_size
        left = self._size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), self._size)
        return self._indices_subset[begin:end]

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

    def _get_indices(self, generator: torch.Generator):
        if self._shuffle:
            shuffle_indices = torch.randperm(self._size_subset, generator=generator)
            indices = self._indices_subset[shuffle_indices].tolist()
        else:
            indices = self._indices_subset.tolist()

        if self._round_up:
            indices = (indices * int(self._padded_size / len(indices) + 1))[
                : self._padded_size
            ]
        return indices

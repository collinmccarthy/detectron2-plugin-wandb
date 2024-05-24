import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data as torchdata
from detectron2.config import configurable
from detectron2.utils.logger import _log_api_usage

from detectron2.data.build import (
    get_detection_dataset_dicts,
    _build_weighted_sampler,
    trivial_batch_collator,
    worker_init_reset_seed,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
    InferenceSampler,
)
from detectron2.data.common import (
    AspectRatioGroupedDataset,
    DatasetFromList,
    MapDataset,
    ToIterableDataset,
)
from detectron2.utils.comm import get_world_size

from .distributed_sampler import (
    EpochTrainingSampler,
    RandomSubsetEpochTrainingSampler,
    RandomSubsetInferenceSampler,
)
from .config import get_config_value

logger = logging.getLogger(__name__)


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    """
    Same as detectron2.data.build._train_loader_from_config but adds "EpochTrainingSampler"
    """
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=(
                cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0
            ),
            proposal_files=(
                cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
            ),
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "EpochTrainingSampler":  # Added
                sampler = EpochTrainingSampler(
                    dataset_size=len(dataset), total_batch_size=total_batch_size
                )
            elif sampler_name == "RandomSubsetEpochTrainingSampler":  # Added
                sampler = RandomSubsetEpochTrainingSampler(
                    dataset_size=len(dataset),
                    total_batch_size=total_batch_size,
                    subset_ratio=cfg.DATALOADER.TRAIN_RANDOM_SUBSET_RATIO,
                    subset_size=cfg.DATALOADER.TRAIN_RANDOM_SUBSET_SIZE,
                )
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset,
                    cfg.DATALOADER.REPEAT_THRESHOLD,
                    sqrt=cfg.DATALOADER.REPEAT_SQRT,
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors, seed=cfg.SEED)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
                )
            elif sampler_name == "WeightedTrainingSampler":
                sampler = _build_weighted_sampler(cfg)
            elif sampler_name == "WeightedCategoryTrainingSampler":
                sampler = _build_weighted_sampler(cfg, enable_category_balance=True)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": total_batch_size,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "persistent_workers": cfg.DATALOADER.PERSISTENT_WORKERS,
        "pin_memory": cfg.DATALOADER.PIN_MEMORY,
    }


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=(
            [
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None
        ),
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)

    test_subset_ratio = get_config_value(cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_RATIO")
    test_subset_size = get_config_value(cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_SIZE")
    if test_subset_ratio is not None or test_subset_size is not None:
        if isinstance(dataset, torchdata.IterableDataset):
            raise RuntimeError(
                f"Found cfg.DATALOADER.TEST_RANDOM_SUBSET_RATIO={test_subset_ratio} and"
                f" cfg.DATALOADER.TEST_RANDOM_SUBSET_SIZE={test_subset_size}. Both must be None"
                f" (default if not set) when using IterableDataset."
            )
        sampler = RandomSubsetInferenceSampler(
            size=len(dataset),
            subset_ratio=test_subset_ratio,
            subset_size=test_subset_size,
            seed=cfg.get("SEED", 2025),
        )
    else:
        sampler = (
            InferenceSampler(len(dataset))
            if not isinstance(dataset, torchdata.IterableDataset)
            else None
        )

    return {
        "dataset": dataset,
        "mapper": mapper,
        "sampler": sampler,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "persistent_workers": cfg.DATALOADER.PERSISTENT_WORKERS,
        "pin_memory": cfg.DATALOADER.PIN_MEMORY,
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    **kwargs,
):
    """
    Same as detectron2.data.build.build_detection_train_loader but with different decorator

    We also add persistent workers and pin memory to aspect ratio use case.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    **kwargs,  # Added
) -> torchdata.DataLoader:
    """
    Same as detectron2.data.build.build_detection_test_loader but with different decorator.

    We also add persistent workers and pin memory (via kwargs).
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,  # Added
    )


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = True,
    single_gpu_batch_size=None,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    seed=None,
    **kwargs,
):
    """
    Same as detectron2.data.build_batch_data_loader() but passes in persistent_workers and
    pin_memory for both aspect_ratio_grouping == True and False
    """
    if single_gpu_batch_size:
        if total_batch_size:
            raise ValueError(
                """total_batch_size and single_gpu_batch_size are mutually incompatible.
                Please specify only one. """
            )
        batch_size = single_gpu_batch_size
    else:
        world_size = get_world_size()
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size
    logger = logging.getLogger(__name__)
    logger.info("Making batched data loader with batch_size=%d", batch_size)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler, shard_chunk_size=batch_size)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if aspect_ratio_grouping:
        assert drop_last, "Aspect ratio grouping will drop incomplete batches."
        data_loader = torchdata.DataLoader(
            dataset,
            batch_size=1,  # Handled by AspectRatioGroupedDataset (uses batch_size input)
            drop_last=False,  # Handled by AspectRatioGroupedDataset (uses drop_last=True)
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            generator=generator,
            **kwargs,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            generator=generator,
            **kwargs,
        )

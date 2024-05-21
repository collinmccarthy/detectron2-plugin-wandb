import logging
import torch.utils.data as torchdata

from detectron2.config import configurable
from detectron2.utils.logger import _log_api_usage

from detectron2.data.build import (
    get_detection_dataset_dicts,
    _build_weighted_sampler,
)
from detectron2.data.build import (
    build_detection_train_loader as d2_build_detection_train_loader,
)
from detectron2.data.build import (
    build_detection_test_loader as d2_build_detection_test_loader,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
    InferenceSampler,
)

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
                cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0
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
                repeat_factors = (
                    RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                        dataset,
                        cfg.DATALOADER.REPEAT_THRESHOLD,
                        sqrt=cfg.DATALOADER.REPEAT_SQRT,
                    )
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

    test_subset_ratio = get_config_value(
        cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_RATIO"
    )
    test_subset_size = get_config_value(
        cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_SIZE"
    )
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
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": sampler,
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(*args, **kwargs):
    """
    Same as detectron2.data.build.build_detection_train_loader but with different decorator
    """
    return d2_build_detection_train_loader(*args, **kwargs)


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(*args, **kwargs):
    """
    Same as detectron2.data.build.build_detection_test_loader but with different decorator
    """
    return d2_build_detection_test_loader(*args, **kwargs)

import os
import weakref
import logging
import shutil
from pathlib import Path
from argparse import Namespace

import detectron2.utils.comm as comm
from detectron2.engine.defaults import create_ddp_model, DefaultTrainer, TrainerBase
from detectron2.engine.hooks import PeriodicWriter, PeriodicCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import DatasetEvaluators, COCOPanopticEvaluator, verify_results
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage
from detectron2.config import CfgNode
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
)

from .events import CustomWandbWriter, CustomJSONWriter, CustomCommonMetricPrinter
from .config import (
    update_config_epochs,
    get_config_value,
)
from .train_loop import (
    CustomAMPTrainer,
    CustomSimpleTrainer,
    parse_dataloader,
)
from .build import (
    build_detection_train_loader,
    build_detection_test_loader,
)
from .panoptic_evaluation import PartialCOCOPanopticEvaluator

logger = logging.getLogger(__name__)


class CustomTrainerMixin:
    def __init__(self, cfg):
        """Same as train_net.py but uses CustomAMPTrainer instead of AMPTrainer and
        CustomSimpleTrainer instead of SimpleTrainer. Intended to be used as a mixin like:

        ```
        class CustomTrainer(CustomTrainerMixin, Trainer):
            pass
        ```
        """
        # This init overrides DefaultTrainer.__init__(), so call TrainerBase.__init__() directly
        TrainerBase.__init__(self)
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Update: Build data_loader before model/optimizer so we can set SOLVER.MAX_ITER
        # The data_loader should not depend on the model or optimizer
        data_loader = self.build_train_loader(cfg)

        # Update: Convert epoch-based to iter-based metrics
        steps_per_epoch, batch_size = parse_dataloader(data_loader)
        update_config_epochs(cfg=cfg, steps_per_epoch=steps_per_epoch)

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        model = create_ddp_model(model, broadcast_buffers=False)

        # Update: Use CustomAMPTrainer and CustomSimpleTrainer to log epoch / epoch_float
        trainer_cls = CustomAMPTrainer if cfg.SOLVER.AMP.ENABLED else CustomSimpleTrainer
        self._trainer = trainer_cls(
            steps_per_epoch=steps_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
        )

        # Update: Add early_exit_iter and other metrics we print out in self._train()
        early_exit_iter = get_config_value(cfg=cfg, key="SOLVER.EARLY_EXIT_ITER")
        early_exit_epochs = get_config_value(cfg=cfg, key="SOLVER.EARLY_EXIT_EPOCHS")
        if early_exit_iter is not None and early_exit_epochs is not None:
            raise RuntimeError(
                f"Found both SOLVER.EARLY_EXIT_ITER and SOLVER.EARLY_EXIT_EPOCHS. Expected only one."
            )
        elif early_exit_epochs is not None:
            early_exit_iter = early_exit_epochs * steps_per_epoch
        self._early_exit_iter = early_exit_iter
        self._steps_per_epoch = steps_per_epoch
        self._per_gpu_batch_size = batch_size
        self._total_batch_size = batch_size * comm.get_world_size()

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_writers(self):
        """Same as OneFormer.train_net.py but uses cfg.LOGGER.INTERVAL and cfg.WANDB.ENABLED"""
        log_interval = self.cfg.get("LOGGER", {}).get("INTERVAL", 20)
        json_file = os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CustomCommonMetricPrinter(max_iter=self.max_iter, window_size=log_interval),
            CustomJSONWriter(json_file=json_file, window_size=log_interval),
            CustomWandbWriter(enabled=self.cfg.WANDB.ENABLED),
        ]

    def build_hooks(self):
        """
        Update PeriodWriter hook with log interval, and update PeriodCheckpointer with max_to_keep
        """
        hooks = super().build_hooks()
        log_interval = get_config_value(cfg=self.cfg, key="LOGGER.INTERVAL")
        checkpoint_max_keep = get_config_value(cfg=self.cfg, key="SOLVER.CHECKPOINT_MAX_KEEP")

        for hook in hooks:
            if isinstance(hook, PeriodicWriter) and log_interval is not None:
                hook._period = log_interval
            elif isinstance(hook, PeriodicCheckpointer) and checkpoint_max_keep is not None:
                if not float(checkpoint_max_keep).is_integer() and int(checkpoint_max_keep) >= 1:
                    raise RuntimeError(
                        f"Expected SOLVER.CHECKPOINT_MAX_KEEP to be an integer >= 1,"
                        f" found SOLVER.CHECKPOINT_MAX_KEEP={checkpoint_max_keep}"
                    )
                hook.max_to_keep = checkpoint_max_keep
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        """Same as Trainer.build_train_loader but uses our build_detection_train_loader which
        supports cfg.DATALOADER.SAMPLER_TRAIN == "EpochTrainingSampler" or
        "RandomSubsetEpochTrainingSampler".
        """
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Use our own build_detection_test_loader to support RandomSubsetInferenceSampler"""
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        evaluators: DatasetEvaluators = super().build_evaluator(
            cfg=cfg, dataset_name=dataset_name, output_folder=output_folder
        )

        # Replace COCOPanopticEvaluator with ParitalCOCOPanopticEvaluator if using test subset
        # This is required to avoid a runtime error that will prevent PQ from being calculated
        test_subset_ratio = get_config_value(cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_RATIO")
        test_subset_size = get_config_value(cfg=cfg, key="DATALOADER.TEST_RANDOM_SUBSET_SIZE")
        if test_subset_ratio is not None or test_subset_size is not None:
            if isinstance(evaluators, DatasetEvaluators):  # Multiple evaluators
                for idx, evaluator in enumerate(evaluators._evaluators):
                    if type(evaluator) == COCOPanopticEvaluator:
                        evaluators._evaluators[idx] = PartialCOCOPanopticEvaluator(
                            dataset_name=dataset_name, output_dir=evaluator._output_dir
                        )
            elif type(evaluators) == COCOPanopticEvaluator:  # Single evaluator
                evaluators = PartialCOCOPanopticEvaluator(
                    dataset_name=dataset_name, output_dir=evaluator._output_dir
                )

        return evaluators

    def train(self):
        """Same as DefaultTrainer.train() but calls our _train() instead of super().train()"""
        self._train()
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def _train(self):
        """Same as TrainerBase.train() but supports self._early_exit_iter"""
        start_iter = self.start_iter
        max_iter = self.max_iter
        self.iter = start_iter
        self.max_iter = max_iter

        # Use early_exit_iter only for range in loop below, not for self.max_iter
        # We only want to break out early, don't want to impact any other mechanisms
        if self._early_exit_iter is not None:
            max_iter = min(max_iter, self._early_exit_iter)

        logger.info(
            f"Starting training with start_iter={start_iter}, max_iter={max_iter},"
            f" steps_per_epoch={self._steps_per_epoch},"
            f" per_gpu_batch_size={self._per_gpu_batch_size},"
            f" total_batch_size={self._total_batch_size}"
        )

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


def setup_loggers(cfg: CfgNode) -> None:
    # Update: Setup additional logger for detectron2_plugin and this script
    for name, abbrev in [
        ("mask2former", "mask2former"),  # Originally only this
        ("detectron2_plugin", "d2_plugin"),
        ("__main__", "train_net_custom"),
    ]:
        plugin_logger = setup_logger(
            output=cfg.OUTPUT_DIR,
            distributed_rank=comm.get_rank(),
            name=name,
            abbrev_name=abbrev,
        )
        plugin_logger.setLevel(logging.INFO)
        for handler in plugin_logger.handlers:
            handler.setLevel(logging.INFO)


def maybe_restart_run(args: Namespace, cfg: CfgNode):
    if cfg.get("RESTART_RUN", False):
        if comm.is_main_process():
            args.resume = False  # Don't resume
            remove_dir_names = ["inference", "wandb"]
            remove_file_suffixes = [".txt", ".json"]
            for filepath in Path(cfg.OUTPUT_DIR).glob("*"):  # Cleanup dir
                try:
                    if filepath.is_dir() and filepath.name in remove_dir_names:
                        shutil.rmtree(filepath, ignore_errors=True)
                    elif filepath.is_file() and filepath.suffix in remove_file_suffixes:
                        filepath.unlink()
                except Exception:
                    # Ignore any I/O errors when removing, it's okay if we can't cleanup
                    pass

        comm.synchronize()

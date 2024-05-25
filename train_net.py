import os
import re
import weakref
import logging
import shutil
import time
import signal
from pathlib import Path
from argparse import Namespace
from typing import Callable, Dict, Optional, Set

import torch
from torch.utils.data import DataLoader
from fvcore.nn.precise_bn import get_bn_modules
import detectron2.utils.comm as comm
from detectron2.engine.defaults import create_ddp_model, DefaultTrainer, TrainerBase
from detectron2.engine.defaults import hooks as d2_hooks
from detectron2.engine.hooks import PeriodicWriter, PeriodicCheckpointer, LRScheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.evaluation import COCOPanopticEvaluator
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

from .events import CustomJSONWriter, CustomCommonMetricPrinter
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
from .hooks import CustomLRScheduler, CustomBestCheckpointer
from .wandb import CustomWandbWriter

logger = logging.getLogger(__name__)


class CustomTrainerMixin:
    _dataloaders: Dict[str, DataLoader] = {}

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
        pytorch_data_loader, steps_per_epoch, batch_size = parse_dataloader(data_loader)
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

        self._optimizer_named_params: Optional[Dict[str, Dict]] = None
        CustomTrainerMixin._dataloaders["train"] = pytorch_data_loader

    @property
    def optimizer_named_params(self) -> Dict[str, dict]:
        if self._optimizer_named_params is None:
            # Feature still doesn't exist, see https://github.com/pytorch/pytorch/issues/1489
            # Iterate over named params just like mask2former.Trainer.build_optimizer()
            param_idx = 0
            optimizer_named_params = {}
            for module_name, module in self._trainer.model.named_modules():
                for module_param_name, value in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{module_param_name}"
                    params_dict = self._trainer.optimizer.param_groups[param_idx]
                    assert len(params_dict["params"]) == 1 and id(params_dict["params"][0]) == id(
                        value
                    ), "Params and module tensor mismatch"

                    optimizer_named_params[full_name] = params_dict  # Store orig params dict
                    param_idx += 1

            assert len(optimizer_named_params) == len(
                self._trainer.optimizer.param_groups
            ), "Missing named params"
            self._optimizer_named_params = optimizer_named_params

        return self._optimizer_named_params

    def compute_unique_lr_groups(self) -> Dict[str, float]:
        """
        Iterate over optimizer params and return a dictionary of top-most keys and their current LRs

        This is re-computed each iteration so the LR values are "current"
        """
        # Construct map for lr to named params
        groups_to_lrs: Dict[str, Set[str]] = {}
        for name, params in self.optimizer_named_params.items():
            lr_str = str(params["lr"])  # Use string as key
            split_name = name.split(".")
            for idx in range(len(split_name)):
                group_name = ".".join(split_name[: idx + 1])
                if group_name not in groups_to_lrs:
                    groups_to_lrs[group_name] = set()
                groups_to_lrs[group_name].add(lr_str)

        # Mark names to be removed if they are redundant (the "parent" prefix has only one lr value)
        remove_names = []
        for name in list(groups_to_lrs.keys()):
            parent_name = ".".join(name.split(".")[:-1])
            if parent_name in groups_to_lrs and len(groups_to_lrs[parent_name]) == 1:
                remove_names.append(name)  # This param is "covered" by parent

        unique_lr_groups: Dict[str, float] = {
            name: float(next(iter(lr_vals)))  # Use next(iter(s)) to get single element in set s
            for name, lr_vals in groups_to_lrs.items()
            if name not in remove_names and len(lr_vals) == 1
        }
        return unique_lr_groups  # e.g. {'backbone': 1e-05, 'sem_seg_head': 0.0001}

    def resume_or_load(self, resume=True):
        """
        Same as detectron2 but we update cfg.MODEL.WEIGHTS if it's missing, but found in cache dir
        """
        weights_path = self.cfg.MODEL.WEIGHTS
        if "://" not in self.cfg.MODEL.WEIGHTS:  # Local filepath, try to find in torch cache dir
            weights_path = Path(weights_path)
            if not weights_path.exists():
                cache_dir = torch.hub.get_dir()
                matching_files = [
                    filepath
                    for filepath in Path(cache_dir).rglob(f"*{weights_path.suffix}")
                    if filepath.name == weights_path.name
                ]
                if len(matching_files) == 1:
                    logger.info(
                        f"Weights file {weights_path} not found. Found one matching filename in"
                        f" torchhub cache dir: {matching_files[0]}. Using matching filepath."
                    )
                    weights_path = str(matching_files[0])
                elif not resume:
                    raise RuntimeError(
                        f"Failed to find weights file {weights_path}. Tried to use torchhub cache dir,"
                        f" found {len(matching_files)} in cache dir (must find exactly one to use"
                        f" cache dir path)"
                    )

        # Same as detectron2.DefaultTrainer now
        self.checkpointer.resume_or_load(weights_path, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_writers(self):
        """Same as OneFormer.train_net.py but uses cfg.LOGGER.INTERVAL and cfg.WANDB.ENABLED"""
        log_interval = self.cfg.get("LOGGER", {}).get("INTERVAL", 20)
        json_file = os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")
        return [
            CustomCommonMetricPrinter(
                early_exit_iter=self._early_exit_iter,
                max_iter=self.max_iter,
                window_size=log_interval,
            ),
            CustomJSONWriter(
                steps_per_epoch=self._trainer.steps_per_epoch,  # For EventWriterMixin
                json_file=json_file,
                window_size=log_interval,
            ),
            # Initialize wandb via `setup_wandb` from self._train (only want to init for training)
            CustomWandbWriter(
                steps_per_epoch=self._trainer.steps_per_epoch  # For EventWriterMixin
            ),
        ]

    def build_hooks(self):
        """Same as detectron2 DefaultTrainer.build_hooks with a few additions:
        - Update LRScheduler -> CustomLRScheduler
        - Update PeriodicWriter -> PeriodicWriter with log interval
        - Add BestCheckpointer after EvalHook
        """
        log_interval = get_config_value(cfg=self.cfg, key="LOGGER.INTERVAL", default=20)
        checkpoint_max_keep = get_config_value(cfg=self.cfg, key="SOLVER.CHECKPOINT_MAX_KEEP")
        if (
            checkpoint_max_keep is not None
            and not float(checkpoint_max_keep).is_integer()
            and int(checkpoint_max_keep) >= 1
        ):
            raise RuntimeError(
                f"Expected SOLVER.CHECKPOINT_MAX_KEEP to be an integer >= 1,"
                f" found SOLVER.CHECKPOINT_MAX_KEEP={checkpoint_max_keep}"
            )

        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            d2_hooks.IterationTimer(),
            CustomLRScheduler(),  # Updated: CustomLRScheduler
            (
                d2_hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():  # Updated: Add checkpoint_max_keep
            ret.append(
                d2_hooks.PeriodicCheckpointer(
                    checkpointer=self.checkpointer,
                    period=cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_to_keep=checkpoint_max_keep,
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(d2_hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # Updated: Add BestCheckpointer
        best_metrics = cfg.SOLVER.CHECKPOINT_BEST_METRICS
        wandb_save = cfg.SOLVER.CHECKPOINT_BEST_METRICS_WANDB_SAVE
        if type(best_metrics) != type(wandb_save):
            raise RuntimeError(
                f"Found type(cfg.SOLVER.CHECKPOINT_BEST_METRICS) = {type(best_metrics)} and"
                f" type(cfg.SOLVER.CHECKPOINT_BEST_METRICS_WANDB_SAVE) = {type(wandb_save)}."
                f" Types must match."
            )

        if isinstance(best_metrics, str):
            val_metrics = [best_metrics]
            save_wandb_flags = [wandb_save]
        elif isinstance(best_metrics, (list, tuple)):
            if len(wandb_save) != len(best_metrics):
                raise RuntimeError(
                    f"Found len(cfg.SOLVER.CHECKPOINT_BEST_METRICS) = {len(best_metrics)} and"
                    f" len(cfg.SOLVER.CHECKPOINT_BEST_METRICS_WANDB_SAVE) = {len(wandb_save)}."
                    f" Lenghts must be the same."
                )
            val_metrics = best_metrics
            save_wandb_flags = wandb_save
        elif best_metrics is None:
            assert wandb_save is not None, "Types should match"
            val_metrics = []
            save_wandb_flags = []

        for val_metric, save_wandb in zip(val_metrics, save_wandb_flags):
            ret.append(
                CustomBestCheckpointer(
                    eval_period=cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric=val_metric,
                    save_wandb=save_wandb,
                )
            )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            # Update: Add log_interval
            ret.append(d2_hooks.PeriodicWriter(self.build_writers(), period=log_interval))
        return ret

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
        data_loader = build_detection_test_loader(cfg, dataset_name)
        pytorch_data_loader, _steps_per_epoch, _batch_size = parse_dataloader(data_loader)
        cls._dataloaders["test"] = pytorch_data_loader
        return data_loader  # Original dataloader, not inner pytorch dataloader

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        results = super().test(cfg=cfg, model=model, evaluators=evaluators)
        cls._dataloaders.pop("test")
        return results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        evaluators: DatasetEvaluators = super().build_evaluator(
            cfg=cfg, dataset_name=dataset_name, output_folder=output_folder
        )

        # Replace COCOPanopticEvaluator with ParitalCOCOPanopticEvaluator if using test subset
        # This is required to avoid a runtime error that will prevent PQ from being calculated
        # And we also update how the multiprocessing pool is closed for multi-core PQ results
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
        """Same as DefaultTrainer.train() but calls our _train() instead of super().train().
        Also handles SIGTERM to close Wandb and mark run as preempting (b/c we cancelled it)"""
        self._train()

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def _train(self):
        """Same as TrainerBase.train() but uses a sigterm handler to close wandb correctly, and
        supports self._early_exit_iter"""

        orig_signal_handlers: dict[int, Callable] = {}

        def _sigterm_handler(signal_num, _frame):
            signal_descr: Optional[str] = signal.strsignal(signal_num)

            # Need exit_code=1 when using preempting=True to show 'preempted'
            exit_code = 1
            preempting = True

            # If print statement was interrupted by SIGTERM, can't print to console during handling
            # Use "signal_safe" logger, from CustomRunner.build_logger(), only prints to file
            # See https://stackoverflow.com/questions/45680378/how-to-explain-the-reentrant-runtimeerror-caused-by-printing-in-signal-handlers
            # and https://stackoverflow.com/questions/64147017/logging-signals-in-python
            logger = logging.getLogger("signal_safe")  # Hard-coded in CustomRunner.build_logger()
            logger.info(
                f"Caught signal {signal_descr} ({signal_num}) during training."
                f" Closing wandb backend with exit_code={exit_code}, preempting={preempting}."
            )

            # Use quiet=True for same reasons as signal_save loggger
            # Wandb still not completely silent but the minimal console output doesn't cause issues
            CustomWandbWriter.close_wandb(
                exit_code=exit_code,
                preempting=preempting,
                quiet=True,
                dataloaders=list(CustomTrainerMixin._dataloaders.values()),
            )

            logger.info("Signal handling finished. Re-raising signal with default signal handler")
            signal.signal(signal_num, orig_signal_handlers[signal_num])
            signal.raise_signal(signal_num)

        if comm.is_main_process():
            orig_signal_handlers[signal.SIGCONT] = signal.signal(signal.SIGCONT, _sigterm_handler)
            orig_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _sigterm_handler)

            def _register_prev_handlers():
                signal.signal(signal.SIGCONT, orig_signal_handlers[signal.SIGCONT])
                signal.signal(signal.SIGTERM, orig_signal_handlers[signal.SIGTERM])

            # Child processes such as those in pq_compute_multi_core will all call this handler
            #   but only want this current process to do the cleanup; need to re-register handlers
            # From https://stackoverflow.com/a/74688726/12422298
            os.register_at_fork(after_in_child=lambda: _register_prev_handlers())

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
            f" steps_per_epoch={self._trainer.steps_per_epoch},"
            f" per_gpu_batch_size={self._per_gpu_batch_size},"
            f" total_batch_size={self._total_batch_size}"
        )

        success = False
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
                success = True
            except Exception as e:
                # Sleep for a few seconds and see if we get SIGTERM, if we do the exception came
                #   from a timeout, which triggered a dataloader crash or something
                time.sleep(5)

                # If we reach this point, it was a real exception, close with exit_code=1
                msg = f": {str(e)}" if len(str(e)) > 0 else ""
                logger.info(
                    f"Caught {type(e).__name__}{msg}. Closing wandb (exit_code=1) and re-raising."
                )
                CustomWandbWriter.close_wandb(exit_code=1)
                raise
            finally:
                if success:
                    logger.info(
                        f"Succesfully finished training for {max_iter} iter. Calling after_train()."
                    )
                self.after_train()  # Calls CustomWandbWriter.close() to close wandb

        comm.synchronize()  # If non-main process leaves early, torchrun may terminate main


def setup_loggers(cfg: CfgNode) -> None:
    # Update: Setup additional logger for detectron2_plugin and this script, and a 'signal_safe'
    #   version which can be safely called during SIGTERM handling (can't print to stdout)
    for name, abbrev in [
        ("mask2former", "mask2former"),  # Originally only this
        ("detectron2_plugin", "d2_plugin"),
        ("__main__", "train_net_custom"),
        ("signal_safe", "signal_safe"),
    ]:
        plugin_logger = setup_logger(
            output=cfg.OUTPUT_DIR,
            distributed_rank=comm.get_rank(),
            name=name,
            abbrev_name=abbrev,
            configure_stdout=True if name != "signal_safe" else False,
        )
        plugin_logger.setLevel(logging.INFO)
        for handler in plugin_logger.handlers:
            handler.setLevel(logging.INFO)


def maybe_restart_run(args: Namespace, cfg: CfgNode):
    if cfg.get("RESTART_RUN", False):
        args.resume = False  # Don't resume
        if comm.is_main_process():
            # Don't backup `wandb` dir, wandb already initialized with resume=False
            backup_dir_names_regex = ["inference"]
            backup_file_names_regex = [
                r"log\..+",
                r".+\.json",
                r".+\.pth",
                "last_checkpoint",
            ]  # Top-level dir only

            logger.info(
                f"Found cfg.RESTART_RUN=True, backing up directories matching"
                f" {backup_dir_names_regex} and files matching {backup_file_names_regex}"
            )

            backup_dest_dir = Path(cfg.OUTPUT_DIR, "prev_run")
            if backup_dest_dir.exists():
                logger.info(
                    f"Found previous backup dir {backup_dest_dir}. Deleting previous backup."
                )
                shutil.rmtree(backup_dest_dir, ignore_errors=True)  # ignore_errors req if not empty
            backup_dest_dir.mkdir(parents=True, exist_ok=True)

            for filepath in Path(cfg.OUTPUT_DIR).glob("*"):  # Not recursive
                match_dir = any(
                    [
                        re.search(pattern=regex, string=filepath.name) is not None
                        for regex in backup_dir_names_regex
                    ]
                )
                match_file = any(
                    [
                        re.search(pattern=regex, string=filepath.name) is not None
                        for regex in backup_file_names_regex
                    ]
                )
                if (filepath.is_dir() and match_dir) or (filepath.is_file() and match_file):
                    dest_filepath = backup_dest_dir.joinpath(filepath.name)
                    shutil.move(src=str(filepath), dst=str(dest_filepath))

        comm.synchronize()

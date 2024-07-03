import os
import logging
import time
import signal
from typing import Callable, Optional

import torch
from torch.multiprocessing.spawn import ProcessRaisedException
from detectron2.engine import launch as d2_launch
from detectron2.engine.launch import _distributed_worker
import detectron2.utils.comm as comm

from .config import parse_dist_url, parse_args

# Some methods use a "quiet" logger (only logs to file, for SIGTERM handling)
std_logger = logging.getLogger(__name__)
quiet_logger = logging.getLogger("signal_safe")  # See d2_plugin.train_net.py, setup_loggers()


def launch(main_func: Callable) -> None:
    """Modified from mask2former.train_net.py to support torchrun (used for docker containers)
    and if not using torchrun, auto-increment port if in use."""
    args = parse_args()
    local_rank = os.environ.get("LOCAL_RANK", None)

    if local_rank is not None:  # Torchrun workflow
        nproc_per_node = os.environ.get("LOCAL_WORLD_SIZE", None)
        world_size = os.environ.get("WORLD_SIZE", None)
        node_rank = os.environ.get("GROUP_RANK", None)
        dist_addr = os.environ.get("MASTER_ADDR", None)
        dist_port = os.environ.get("MASTER_PORT", None)
        if any([v is None for v in [nproc_per_node, world_size, node_rank, dist_addr, dist_port]]):
            raise RuntimeError(
                f"Missing one or more torchrun environment variables:"
                f"\n  LOCAL_RANK={local_rank}"
                f"\n  LOCAL_WORLD_SIZE={nproc_per_node}"
                f"\n  WORLD_SIZE={world_size}"
                f"\n  GROUP_RANK={node_rank}"
                f"\n  MASTER_ADDR={dist_addr}"
                f"\n  MASTER_PORT={dist_port}"
            )

        std_logger.info("Found torchrun environment variables, calling main() directly")
        _distributed_worker(
            local_rank=int(local_rank),
            main_func=main_func,
            world_size=int(world_size),
            num_gpus_per_machine=int(nproc_per_node),
            machine_rank=int(node_rank),
            dist_url=f"tcp://{dist_addr}:{dist_port}",
            args=(args,),
        )

    else:  # Normal, non-torchrun workflow
        dist_url, dist_port = parse_dist_url(dist_url=args.dist_url, num_machines=args.num_machines)

        attempt = 0
        while attempt < 20:
            try:
                print("Command Line Args:", args)
                d2_launch(
                    main_func,
                    args.num_gpus,
                    num_machines=args.num_machines,
                    machine_rank=args.machine_rank,
                    dist_url=f"{dist_url}:{dist_port}",
                    args=(args,),
                )
                print("Run finished successfully. Exiting.")
                break

            except ProcessRaisedException as e:
                if "Address already in use" in str(e):
                    print(
                        f"Distributed url {dist_url}:{dist_port} is not available."
                        f" Incrementing port and re-trying."
                    )
                    dist_port += 1
                    attempt += 1
                else:
                    raise e

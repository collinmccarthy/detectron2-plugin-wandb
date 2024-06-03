import logging
from functools import wraps
import torch

from detectron2.utils.memory import _ignore_torch_cuda_oom


def get_time_str_from_sec(total_sec: int):
    # From https://stackoverflow.com/a/539360
    days, remainder = divmod(total_sec, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{int(days)}d:{int(hours)}h:{int(minutes)}m:{int(seconds):02d}s"
    elif hours > 0:
        return f"{int(hours)}h:{int(minutes)}m:{int(seconds):02d}s"
    else:
        return f"{int(minutes)}m:{int(seconds):02d}s"


def retry_if_cuda_oom(func, cpu_dtype: torch.dtype = torch.float32):
    """Same as detectron2.utils.memory.py, retry_if_cuda_oom() but uses FP32 for CPU (by default).
    Many operations (e.g. softmax) aren't implemented for FP16 on CPU.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu", dtype=cpu_dtype)
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped

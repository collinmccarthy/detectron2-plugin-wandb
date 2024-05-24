"""Timeout class with signals, based on https://stackoverflow.com/a/22547155"""

import signal
from logging import Logger
from signal import Signals


class Timeout:
    """Raise signal after timeout_sec number of seconds"""

    def __init__(
        self,
        timeout_sec: int,
        timeout_msg: str,
        logger: Logger,
        signal_num: Signals = signal.SIGKILL,
    ):
        # Must use logger="signal_safe" if calling from signal handler
        self.logger = logger
        self.timeout_sec = timeout_sec
        self.raise_signal = signal_num
        self.timeout_msg = timeout_msg

    def __enter__(self, *_args, **_kwargs):
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(self.timeout_sec)

    def handler(self, *_args, **_kwargs):
        self.logger.info(self.timeout_msg)
        signal.raise_signal(self.raise_signal)

    def __exit__(self, *_args, **_kwargs):
        signal.alarm(0)  # Disable alarm

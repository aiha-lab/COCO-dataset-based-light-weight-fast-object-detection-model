import time

import torch

__all__ = ["time_synchronized", "TimeTracker"]


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.process_time_ns()


class TimeTracker(object):

    def __init__(self, profile: bool = True):
        self.profile = profile
        self.t = time_synchronized() if self.profile else 0.0

    def update(self):
        if not self.profile:
            return 0.0
        new_t = time_synchronized()
        duration = new_t - self.t
        self.t = new_t
        return duration / 1e6  # ms

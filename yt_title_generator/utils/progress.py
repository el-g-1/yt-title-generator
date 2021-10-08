import progress.bar as progress_bar
import threading


class Bar:
    '''Progress bar'''

    def __init__(self, text, max):
        self._bar = progress_bar.Bar(
            text, max=max, suffix="%(percent).1f%% (ETA:%(eta)ds) | %(index)d/%(max)d"
        )

    def next(self):
        self._bar.next()

    def finish(self):
        self._bar.finish()


class LockedBar(Bar):
    '''Thread-safe progress bar'''
    def __init__(self, text, max):
        super().__init__(text, max)
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            super().next()

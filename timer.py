from time import time
import contextlib


class Timer:
    def __init__(self):
        self.times:dict[str,float] = {}
        self.counts:dict[str,int] = {}
    @contextlib.contextmanager
    def time(self, key):
        s = time()
        try:
            yield
        finally:
            self.times[key] = self.times.get(key, 0) + time() - s
            self.counts[key] = self.counts.get(key, 0) + 1
    def nice_summary(self):
        print(f"{'name':>25} : {'total':20} : {'avg':20}")
        for key,total_time in self.times.items():
            print(f"{key:>25} : {total_time:<010.10f} : {total_time/self.counts[key]:<010.10f}")
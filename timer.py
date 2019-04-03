import time

from datetime import datetime


class Timer:
    def __init__(self, total_ticks=None):
        self.times = None
        self.last = None
        self.start_time = None
        self.total_ticks = total_ticks

    def reset(self):
        self.__init__(total_ticks=self.total_ticks)

    def start(self):
        self.last = time.time()
        self.start_time = self.last
        self.times = []

    def tick(self):
        now = time.time()
        self.times.append(now - self.last)
        self.last = now

    def get_average(self):
        return sum(self.times) / len(self.times)

    def since_start(self):
        now = time.time()
        t = now - self.start_time
        return hms(t)

    def remaining(self):
        if len(self.times) == 0:
            return ""
        n = self.total_ticks - len(self.times)
        t = self.get_average() * n
        h, m, s = hms(t)
        return f"{h:d}:{m:02d}:{s:02d} remaining"


def hms(t):
    h, t = divmod(int(t), 3600)
    m, s = divmod(t, 60)
    return h, m, s


def timestamp(format="%H:%M:%S"):
    now = datetime.now()
    return now.strftime(format)

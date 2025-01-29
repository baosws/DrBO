import time

class Logger:
    def __init__(self, verbose):
        self.verbose = verbose
        self.track_logs = []
        self.temp_logs = []
        self.start_time = time.perf_counter()
        self._step = 0

    def __call__(self, *str):
        self.temp_logs.extend(str)

    def log_str(self):
        return ','.join(self.temp_logs)
    
    def set_step(self, step):
        self._step = step
    
    def add(self, **kwargs):
        self.track_logs.append({
            'step': self._step,
            'timestamp': time.perf_counter() - self.start_time,
            **kwargs
        })

    def reset(self):
        self.temp_logs = []
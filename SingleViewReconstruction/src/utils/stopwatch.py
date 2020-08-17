import time
from src.utils.timeframe import TimeFrame

class StopWatch(object):

	def __init__(self):
		"""Initialize a new `Stopwatch`, but do not start timing."""
		self.start_time = None
		self.stop_time = None
		self.start()
	
	def start(self):
		"""Start timing."""
		self.start_time = time.time()

	def stop(self):
		"""Stop timing."""
		self.stop_time = time.time()
		
	def reset(self):
		self.start()
		self.stop_time = self.start_time

	@property
	def elapsed_time_val(self):
		"""Returns the elapsed time since start was object created or start was called"""
		return time.time() - self.start_time

	@property
	def elapsed_time(self):
		return str(TimeFrame(time.time() - self.start_time))

	@property
	def total_run_time_val(self):
		return self.stop_time - self.start_time
	
	@property
	def total_run_time(self):
		return str(TimeFrame(self.stop_time - self.start_time))

	def __str__(self):
		return self.elapsed_time


if __name__ == '__main__':
	sw = StopWatch()
	time.sleep(0.2)
	print(sw.elapsed_time)

from src.utils.stopwatch import StopWatch
from src.utils.averagenumber import AverageNumber
from src.utils.timeframe import TimeFrame
import time

class AverageStopWatch(object):

	def __init__(self):
		self._sw = StopWatch()
		self._avg_nr = AverageNumber()
		self._total_sw = StopWatch()

	def start(self):
		self._sw.start()
		self._avg_nr = AverageNumber()
		self._total_sw = StopWatch()
	
	def reset(self):
		self._sw.start()
		self._avg_nr = AverageNumber()

	def reset_time(self):
		time = self._sw.elapsed_time_val
		self._avg_nr.add_new(time)
		self._sw.reset()
		return self.avg_time

	def start_timer(self):
		self._sw.start()

	@property
	def avg_time(self):
		return str(TimeFrame(self._avg_nr.mean))
	
	@property
	def avg_time_val(self):
		return self._avg_nr.mean

	@property
	def counter(self):
		return self._avg_nr.counter

	@property
	def total_run_time(self):
		return self._total_sw.elapsed_time

if __name__ == '__main__':
	avg = AverageStopWatch()
	for i in range(100):
		time.sleep(0.15)
		print(avg.reset_time(), avg.total_run_time)


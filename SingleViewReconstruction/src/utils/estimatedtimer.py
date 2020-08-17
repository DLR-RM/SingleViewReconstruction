from src.utils.averagestopwatch import AverageStopWatch

import time

class EstimatedTimer(object):

	def __init__(self):
		self._sw = AverageStopWatch()

	def done_loop(self):
		self._sw.reset_time()

	@property
	def amount_of_iterations(self):
		return self._sw.counter

	@property
	def average_time(self):
		return self._sw.avg_time

	@property
	def average_time_val(self):
		return self._sw.avg_time_val

	@property
	def total_run_time(self):
		return self._sw.total_run_time

	def estimate_rest_time(self, max_number):
		if self._sw.counter > 0:
			avg_time_val = self.average_time_val 
			counter = self._sw.counter
			return float(max_number - counter) * avg_time_val
		return 0.0

	def reset(self):
		self._sw.reset()

	
if __name__ == '__main__':
	es = EstimatedTimer()
	for i in range(20):
		time.sleep(0.1)
		es.done_loop()
		print(es.average_time, es.total_run_time)
	es = EstimatedTimer()
	for i in range(10):
		time.sleep(0.1)
		es.done_loop()
		print((es.estimate_rest_time(10)))


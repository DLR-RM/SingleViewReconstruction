
class AverageNumber(object):

	def __init__(self, start_value = 0.0):
		self._mean = float(start_value)
		self._counter = 0.0
	
	def add_new(self, value):
		self._counter += 1.0
		fac = 1.0 / self._counter
		self._mean = fac * value + (1.0 - fac) * self._mean
	
	@property
	def mean(self):
		return self._mean

	@property
	def counter(self):
		return self._counter

	def reset(self, start_value = 0.0):
		self._counter = 0
		self._mean = float(start_value)


if __name__ == '__main__':
	avg =	AverageNumber(5)
	print(avg.mean, avg.counter)
	avg.add_new(1.0)
	print(avg.mean, avg.counter)
	avg.add_new(2.0)
	print(avg.mean, avg.counter)


class TimeFrame(object):
	
	_string_repr = [[" day", " hour", " min", " sec"],[" d", " h", " m", " s"]]

	def __init__(self, seconds = 0.0):
		self._seconds = 0.0
		self._minutes = 0
		self._hours = 0
		self._days = 0
		self._use_short = int(False)
		self._total_seconds = seconds
		self.set_with_seconds(seconds)
	
	def use_short_repr(self, use_short = True):
		self._use_short = int(use_short)

	def set_with_seconds(self, seconds):
		self._total_seconds = seconds
		seconds = float(seconds)
		if seconds >= 0.0:
			if seconds > 86400:
				self._days = int(seconds / 86400)
				self._hours = int((seconds - self._days * 86400) / 3600)
				self._minutes = int((seconds - self._days * 86400 - self._hours * 3600) / 60)
				self._seconds = seconds % 60.0
			elif seconds > 3600.0:
				self._hours = int(seconds / 3600)
				self._minutes = int((seconds - self._hours * 3600) / 60)
				self._seconds = seconds % 60.0
			elif seconds > 60.0:
				self._hours = 0
				self._minutes = int(seconds / 60.0)
				self._seconds = seconds % 60.0
			else:
				self._hours, self._minutes = 0, 0
				self._seconds = seconds

	def __str__(self):
		if self._total_seconds < 1.0 and self._total_seconds > 0:
			return "{:1.5f}".format(self._seconds) + TimeFrame._string_repr[self._use_short][3]
		result = ""
		if self._days > 0:
			result += str(self._days) + TimeFrame._string_repr[self._use_short][0]
			if self._use_short == 0 and self._days > 1:
				result += "s"
			if self._hours > 0 or self._minutes > 0 or self._seconds > 0:
				result += " " 
		if self._hours > 0:
			result += str(self._hours) +  TimeFrame._string_repr[self._use_short][1]
			if self._use_short == 0 and self._hours > 1:
				result += "s"
			if self._minutes > 0 or self._seconds > 0:
				result += " "
		if self._minutes > 0:
			result += str(self._minutes) + TimeFrame._string_repr[self._use_short][2]
			if self._seconds > 0:
				result += " " 
		if self._seconds > 0:
			result += "{:2.2f}".format(self._seconds) + TimeFrame._string_repr[self._use_short][3]
		if len(result) == 0:
			result = "0" + TimeFrame._string_repr[self._use_short][3]
		return result

if __name__ == '__main__':
	tf = TimeFrame()
	print(tf)
	tf = TimeFrame(5)
	print(TimeFrame(0.111))
	print(tf)
	tf = TimeFrame(1000)
	print(tf)
	tf = TimeFrame(4002.2)
	print(tf)
	print(TimeFrame(90320.21))
	print(TimeFrame(190402.8))
	

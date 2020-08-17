from src.utils.stopwatch import StopWatch
from src.utils.estimatedtimer import EstimatedTimer
from src.utils.timeframe import TimeFrame
import sys

import time

class TimedPrinter(object):

	def __init__(self):
		self._sw = StopWatch()

	def print(self, *args, **key_words):
		sep = " " 
		if "sep" in key_words:
			sep = key_words["sep"]
		end = "\n"
		if "end" in key_words:
			end = key_words["end"]
		if len(args) == 0:
			print("Timed: " + self._sw.elapsed_time, sep=sep, end=end)
		else:
			print("Timed: " + self._sw.elapsed_time + " |", *args, sep=sep, end=end)


class TimedNamedPrinter(object):

	def __init__(self):
		self._sw = StopWatch()
	
	def print(self, name, *args, **key_words):
		sep = " " 
		if "sep" in key_words:
			sep = key_words["sep"]
		end = "\n"
		if "end" in key_words:
			end = key_words["end"]
		new_name = name[0].upper() + name[1:]
		if len(args) == 0:
			print(new_name + ": " + self._sw.elapsed_time, sep=sep, end=end)
		else:
			print(new_name + ": " + self._sw.elapsed_time + " |", *args, sep=sep, end=end)
		self._sw.start()

class TimedNamedLoopPrinter(object):

	def __init__(self, loop_name, max_iterations, use_filler=True):
		self._es = EstimatedTimer()
		self._loop_name = loop_name
		self._max_iterations = max_iterations
		self._use_filler = use_filler
		self._error_flag = False
	
	def printLoop(self, *args, **key_words):
		self._es.done_loop()
		counter = int(self._es.amount_of_iterations)
		if counter > self._max_iterations:
			self._max_iterations = counter
			self._error_flag = True
		rest_time = self._es.estimate_rest_time(self._max_iterations)
		sep = " " 
		if "sep" in key_words:
			sep = key_words["sep"]
		output = self._loop_name + "(" + str(counter) + "/" + str(self._max_iterations) + "): " 
		if self._use_filler:
			amount_of_blocks = 20
			output += "["
			for i in range(amount_of_blocks):
				if i < int(counter / float(self._max_iterations) * amount_of_blocks):
					output += "="
				elif i == int(counter / float(self._max_iterations) * amount_of_blocks):
					output += ">"
				else:
					output += "."
			output += "] "	
		end = ""
		if counter != self._max_iterations:
			output += "| ETA: " + str(TimeFrame(rest_time))
		else:
			if not self._error_flag:
				end = "\n"
			output += "| EST " + str(self._es.total_run_time)
		if self._error_flag:
			output += " [Counter is incorrect!]"
		if len(args) > 0:
			output += " - " 
		for ele in args:
			output += str(ele) + sep
		if counter == 1:
			print(output, end=end)
		else:
			white_spaces = ""
			for i in range(len(output) + 15):
				white_spaces += " " 
			print("\r" + white_spaces, end="")
			print("\r" + output, end=end)
			sys.stdout.flush()
		

if __name__ == '__main__':
	tf = TimedPrinter()
	tf.print(sep=' ', end='\n')
	tf.print("huhu2")
	tf = TimedNamedPrinter()
	tf.print("Loading done")
	time.sleep(0.3)
	tf.print("loading finished")

	tLoop = TimedNamedLoopPrinter("Loop iteration", 10)
	for i in range(100):
		time.sleep(0.05)
		tLoop.printLoop("huhu",i)


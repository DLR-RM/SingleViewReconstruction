
from src.utils.stopwatch import StopWatch


class StopWatchContext(object):

    def __init__(self, end_text, start_text = None):
        self._text = end_text
        self._start_text = start_text

    def __enter__(self):
        if self._start_text is not None and isinstance(self._start_text, str):
            print(self._start_text)
        self._sw = StopWatch()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self._text, str) and len(self._text) > 0:
            if self._text[-1] == ' ' or self._text[-1] == '\t':
                print(self._text + str(self._sw))
            else:
                print(self._text + ' ' + str(self._sw))



class Bar:
    def __init__(self, date, time, open_value, high_value, low_value, close_value):
        self._date = date
        self._time = time
        self._open = open_value
        self._high = high_value
        self._low = low_value
        self._close = close_value

    @property
    def open(self):
        return self._open

    @property
    def high(self):
        return self._high

    @property
    def low(self):
        return self._low

    @property
    def close(self):
        return self._close

    @property
    def date(self):
        return self._date

    @property
    def time(self):
        return self._time

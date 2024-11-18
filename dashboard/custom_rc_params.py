import contextlib
from collections import deque


class CustomRCParams(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__historic_parameter_set = deque()

    @contextlib.contextmanager
    def rc_context(self, **parameters):
        try:
            old_values = {k: self.get(k, None) for k in parameters}
            self.__historic_parameter_set.append(old_values)
            self.update(parameters)
            yield

        finally:
            restore_values = self.__historic_parameter_set.pop()
            self.update(restore_values)







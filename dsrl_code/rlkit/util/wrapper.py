import os
LOG_DIR = os.getcwd()


class Wrapper(object):
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, attr):
        is_magic = attr.startswith('__') and attr.endswith('__')
        if is_magic:
            return super().__getattr__(attr)
        try:
            return self.__dict__[attr]
        except:
            return getattr(self.inner, attr)


class SimpleWrapper(object):
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, attr):
        if attr == '_inner':
            raise AttributeError()
        return getattr(self._inner, attr)

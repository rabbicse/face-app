class SingletonDecorator(object):
    def __init__(self, klass):
        """
        @param klass:
        """
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        """
        @param args:
        @param kwargs:
        @return:
        """
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance

import functools
import time
import utils.vision_utils.log_utils as log_utils


def timeit(method):
    """
    decorator to measure elapsed time of any function or method
    """

    def timed(*args, **kw):
        """
        wrapped function to execute
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            pass
            # logger.info('=== %r  %2.2f ms ===' % (method.__name__, (te - ts) * 1000))
        print('=== %r  %2.2f ms ===' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class TimeitDecorator(object):
    """
    FPS decorator, it'll write debug log for any function or method wrap by this decorator
    """

    logger = None
    counter = 0
    time_threshold = 1
    sum_time = 0

    def __init__(self, time_threshold=1, fps=False):
        self.time_threshold = time_threshold
        self.fps = fps

    def __call__(self, fn):
        """
        @param fn:
        @return:
        """

        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            """
            @param args:
            @param kwargs:
            @return:
            """
            if TimeitDecorator.logger is None:
                TimeitDecorator.logger = log_utils.LogUtils().get_logger('timeit-decorator')

            result = None
            try:
                # set start time
                ts = time.time()
                result = fn(*args, **kwargs)
                te = time.time()
                time_diff = te - ts
                time_diff_ms = float(time_diff * 1000)

                if 'log_time' in kwargs:
                    name = kwargs.get('log_name', fn.__name__.upper())
                    kwargs['log_time'][name] = time_diff_ms
                else:
                    pass
                    # logger.info('=== %r  %2.2f ms ===' % (method.__name__, (te - ts) * 1000))
                TimeitDecorator.logger.info('=== %r  %2.2f ms ===' % (fn.__name__, time_diff_ms))

                # calculate fps
                if self.fps is True:
                    self.counter += 1
                    self.sum_time += time_diff
                    if self.sum_time >= self.time_threshold:
                        TimeitDecorator.logger.info(
                            'FPS [{}]: {:.2f}'.format(fn.__name__.upper(),
                                                      (self.counter / self.sum_time) * self.time_threshold))
                        self.counter = 0
                        self.sum_time = 0

                return result
            except Exception as ex:
                TimeitDecorator.logger.error("Exception {0}".format(ex))
            return result

        return decorated


class FpsCountDecorator(object):
    """
    FPS count decorator, it'll write debug log for any function or method wrap by this decorator
    """

    logger = None
    counter = 0
    time_threshold = 1
    start_time = None

    def __init__(self, time_threshold=1):
        self.time_threshold = time_threshold

    def __call__(self, fn):
        """
        @param fn:
        @return:
        """

        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            """
            @param args:
            @param kwargs:
            @return:
            """
            if FpsCountDecorator.logger is None:
                FpsCountDecorator.logger = log_utils.LogUtils().get_logger('fps-counter-decorator')

            if FpsCountDecorator.start_time is None:
                FpsCountDecorator.start_time = time.time()

            result = None
            try:
                result = fn(*args, **kwargs)

                FpsCountDecorator.counter += 1
                time_diff = time.time() - FpsCountDecorator.start_time
                if time_diff >= FpsCountDecorator.time_threshold:
                    FpsCountDecorator.logger.info(
                        'FPS [{}]: {}'.format(fn.__name__.upper(),
                                              FpsCountDecorator.counter / FpsCountDecorator.time_threshold))
                    FpsCountDecorator.counter = 0
                    FpsCountDecorator.start_time = time.time()

                return result
            except Exception as ex:
                FpsCountDecorator.logger.error("Exception {0}".format(ex))
            return result

        return decorated


class FpsDecorator(object):
    """
    FPS decorator, it'll write debug log for any function or method wrap by this decorator
    """

    def __init__(self):
        self.logger = log_utils.LogUtils().get_logger('fps-decorator')

    def __call__(self, fn):
        """
        @param fn:
        @return:
        """

        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            """
            @param args:
            @param kwargs:
            @return:
            """
            # set start time
            start_time = time.time()  # start time of the loop
            result = None
            try:
                result = fn(*args, **kwargs)

                # calculate FPS
                t = time.time() - start_time
                fps = 1.0 / t
                self.logger.info('FPS: {}'.format(fps))  # FPS = 1 / time to process loop
                return result
            except Exception as ex:
                self.logger.error("Exception {0}".format(ex))
            return result

        return decorated


class LogDecorator(object):
    """
    Log decorator, it'll write debug log for any function or method wrap by this decorator
    """

    def __init__(self):
        self.logger = log_utils.LogUtils().get_logger('log-decorator')

    def __call__(self, fn):
        """
        @param fn:
        @return:
        """

        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            """
            @param args:
            @param kwargs:
            @return:
            """
            result = None
            try:
                self.logger.debug("{0} - {1} - {2}".format(fn.__name__, args, kwargs))
                result = fn(*args, **kwargs)
                self.logger.debug(result)
                return result
            except Exception as ex:
                self.logger.error("Exception {0}".format(ex))
            return result

        return decorated

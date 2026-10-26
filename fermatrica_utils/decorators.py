"""
Decorators to be used in different parts and contexts
"""


import time
from multiprocessing.pool import ThreadPool
import itertools
import sys
import warnings


def spinner(f):
    def f_(*args
           , **kwargs):
        """
        Spinner for time-consuming functions. Use as a decorator:

        .. code-block:: python
            @spinner
            def my_fun(a, b):
                return a * b
            
            my_fun(3, 4)

        Works with PyCharm Debug as well despite multithreading (although not multiprocessing)

        :param args:
        :param kwargs:
        :return:
        """

        pool = ThreadPool(processes=2)
        t = pool.apply_async(func=f, args=(*args,), kwds=kwargs)

        start_time = time.time()

        spinner_items = u'◜◝◞◟'

        cgreen2 = '\33[92m'
        cyellow2 = '\33[93m'
        cend = '\33[0m'
        erase_line = '\r\33[2K'

        time_sleep = 0.25
        spinner_items = itertools.cycle(spinner_items)

        # the frame animation is a terminal effect: a redirected stdout keeps the finish line alone,
        # so a captured run log stays readable
        if_animate = sys.stdout.isatty()

        showwarning_outer = warnings.showwarning

        def showwarning_(*args_w
                         , **kwargs_w):
            """
            Give every warning a cleared line to start on: the decorated function runs in a worker
            thread and writes to stderr while the current spinner frame is still open.
            """

            sys.stdout.write(erase_line)
            sys.stdout.flush()

            showwarning_outer(*args_w, **kwargs_w)

        if if_animate:
            warnings.showwarning = showwarning_

        try:
            while not t.ready():

                if if_animate:
                    time_sp = str(int(round(time.time() - start_time, 0)))
                    stringout = cyellow2 + "running " + f.__name__ + " " + next(
                        spinner_items) + cend + " current time spent : " + " " + time_sp + " sec"

                    sys.stdout.write(erase_line + stringout)
                    sys.stdout.flush()

                time.sleep(time_sleep)

        finally:
            if if_animate:
                warnings.showwarning = showwarning_outer
                sys.stdout.write(erase_line)
                sys.stdout.flush()

        time_sp = str(round(time.time() - start_time, 2))
        print(cgreen2 + f.__name__ + " is finished " + u'\u2713' + cend + " total time spent - " + time_sp + " sec")

        return t.get()

    return f_


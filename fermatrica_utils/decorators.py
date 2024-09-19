"""
Decorators to be used in different parts and contexts
"""


import time
from multiprocessing.pool import ThreadPool
import itertools
import sys


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

        time_sleep = 0.25
        spinner_items = itertools.cycle(spinner_items)

        while not t.ready():

            time_sp = str(int(round(time.time() - start_time, 0)))
            stringout = cyellow2 + "running " + f.__name__ + " " + next(
                spinner_items) + cend + " current time spent : " + " " + time_sp + " sec"

            sys.stdout.write(stringout)
            sys.stdout.flush()

            time.sleep(time_sleep)
            sys.stdout.write("\r")

        time_sp = str(round(time.time() - start_time, 2))
        print(cgreen2 + f.__name__ + " is finished " + u'\u2713' + cend + " total time spent - " + time_sp + " sec")

        return t.get()

    return f_


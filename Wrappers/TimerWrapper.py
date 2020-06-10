# -*- coding: utf-8 -*-

import time


class TimerWrapper(object):
    def _init_(decorated):
        self._decorated = decorated

    def Timer(printText):
        def wrap(fun):
            def wrapped_function(*args,**kwargs):
                start_time = time.time()
                ret = fun(*args, **kwargs)

                print('{:>6.2f} s | '.format(time.time() - start_time),end='')
                print('function: ',printText,'()',sep='', end='')
                print('*args: ',*args[1:])

                return ret
            return wrapped_function
        return wrap
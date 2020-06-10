# -*- coding: utf-8 -*-
"""
source: https://stackoverflow.com/a/5849861

how to use:
    
with Timer('foo_stuff'):
   # do some stuff
    

"""

import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: {:02.2f} s'.format(time.time() - self.tstart))
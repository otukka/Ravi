# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:52:52 2020

@author: VStore
"""

from Handlers.DailyWebCrawlerHandler import DailyWebCrawlerHandler

import sys
import os
import re
import logging
import traceback
import tkinter
from tkinter.messagebox import showerror

def main(argv):
    DWCH = DailyWebCrawlerHandler()
    DWCH.fetch_results()
    DWCH.save_gzip()    


if __name__ == '__main__':


    # Logging initialization filename.log
    filename = os.path.basename(sys.argv[0])
    filename = filename + '.log'
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Remove excess logging form request library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:

        main(sys.argv[1:])

    except Exception as e:

        lines = traceback.format_exc().splitlines()
        l = re.search('(line [\d]*)', lines[-3])
        f = re.search('([\w_]*.py)', lines[-3])


        # Error message
        if l:
            lno = l.group(1)
        else:
            lno = "regex parse error"

        if f:
            filename = f.group(1)
        else:
            filename = "regex parse error"
            
        err = lines[-1].rstrip().lstrip()
        line = lines[-2].rstrip().lstrip()        
        msg = 'Error in '+filename+' '+lno+' => '+line+ ' / '+err+'\n'


        logging.exception(msg, exc_info=False)
        root = tkinter.Tk()
        root.withdraw()
        showerror(title = "Error", message =msg)
        root.destroy()

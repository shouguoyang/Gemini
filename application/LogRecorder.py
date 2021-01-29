# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LogRecorder
   Description :
   Author :       ysg
   date：          2021/1/29
-------------------------------------------------
   Change Activity:
                   2021/1/29:
-------------------------------------------------
"""

import logging


class CLogRecoder:

    def __init__(self, logfile = 'log.log', format = '%(asctime)s : %(message)s', level = logging.DEBUG):
        logging.basicConfig(filename= logfile, level= level , format= format)
        self._ft = format

    def addStreamHandler(self):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formater = logging.Formatter(self._ft)
        console.setFormatter(formater)
        logging.getLogger('').addHandler(console)
        return self

    def INFO(self, message):
        logging.info(message)
        return self



if __name__ == '__main__':
    lr = CLogRecoder().addStreamHandler()
    lr.INFO("test")

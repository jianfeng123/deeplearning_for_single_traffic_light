import logging
import os
from collections import defaultdict
from colorlog import ColoredFormatter

import pandas
import numpy as np
from tabulate import tabulate

class LoggerClass(object):
    GLOBAL_LOGGER_NAME = '_global_logger'

    _color_formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(name)-10s %(levelname)-8s%(reset)s %(white)s%(message)s",
        datefmt='%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    _normal_formatter = logging.Formatter(
        '%(asctime)s %(name)-10s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        style='%'
    )
    def __init__(self):
        self._dir = None
        self._logger = None
        self._log_path = None
        self._tabular = defaultdict(list)
        self._curr_recorded = list()
        self._num_dump_tabular_calls = 0

    @property
    def dir(self):
        return self._dir

    #############
    ### Setup ###
    #############

    def setup(self, display_name, log_path, lvl):
        self._dir = os.path.dirname(log_path)
        self._logger = self._get_logger(LoggerClass.GLOBAL_LOGGER_NAME,
                                        log_path,
                                        lvl=lvl,
                                        display_name=display_name)

    def _get_logger(self, name, log_path, lvl=logging.INFO, display_name=None):
        if isinstance(lvl, str):
            lvl = lvl.lower().strip()
            if lvl == 'debug':
                lvl = logging.DEBUG
            elif lvl == 'info':
                lvl = logging.INFO
            elif lvl == 'warn' or lvl == 'warning':
                lvl = logging.WARN
            elif lvl == 'error':
                lvl = logging.ERROR
            elif lvl == 'fatal' or lvl == 'critical':
                lvl = logging.CRITICAL
            else:
                raise ValueError('unknown logging level')

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LoggerClass._normal_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(lvl)
        console_handler.setFormatter(LoggerClass._color_formatter)
        if display_name is None:
            display_name = name
        logger = logging.getLogger(display_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    ###############
    ### Logging ###
    ###############

    def debug(self, s):
        assert (self._logger is not None)
        self._logger.debug(s)

    def info(self, s):
        assert (self._logger is not None)
        self._logger.info(s)

    def warn(self, s):
        assert (self._logger is not None)
        self._logger.warn(s)

    def error(self, s):
        assert (self._logger is not None)
        self._logger.error(s)

    def critical(self, s):
        assert (self._logger is not None)
        self._logger.critical(s)


logger = LoggerClass()



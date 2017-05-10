from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import cv2
from collections import namedtuple

def loggerConfig(log_file, verbose=2):
   logger      = logging.getLogger()
   formatter   = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
   fileHandler = logging.FileHandler(log_file, 'w')
   fileHandler.setFormatter(formatter)
   logger.addHandler(fileHandler)
   if verbose >= 2:
       logger.setLevel(logging.DEBUG)
   elif verbose >= 1:
       logger.setLevel(logging.INFO)
   else:
       # NOTE: we currently use this level to log to get rid of visdom's info printouts
       logger.setLevel(logging.WARNING)
   return logger

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
# NOTE: used as the return format for Env(), and for format to push into replay memory for off-policy methods
# NOTE: when return from Env(), state0 is always None
Experience  = namedtuple('Experience',  'state0, action, reward, state1, terminal1')

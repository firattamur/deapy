import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from abc import ABC, abstractmethod
from utils.enums import Optimizer


class AbstractDEAOptimizer(ABC):


    @abstractmethod
    def __init__(self, optimizer: Optimizer = Optimizer.GLPK, time_limit: float = None, silent: bool = True):
        self.optimizer  : Optimizer = optimizer
        self.time_limit : int       = time_limit
        self.silent     : bool      = silent

from abc import ABC, abstractmethod
from dea.utils.symbols import Optimizer


class AbstractDEAOptimizer(ABC):


    @abstractmethod
    def __init__(self, optimizer: Optimizer = Optimizer.GLPK, time_limit: float = None, silent: bool = True):
        self.optimizer  : Optimizer = optimizer
        self.time_limit : int       = time_limit
        self.silent     : bool      = silent

from src.core.abstract_dea_optimizer import AbstractDEAOptimizer
from src.core.symbols import Optimizer


class DEAOptimizer(AbstractDEAOptimizer):

    def __init__(self, optimizer: Optimizer = Optimizer.GLPK, time_limit: float = None, silent: bool = True):
        super(DEAOptimizer, self).__init__(optimizer, time_limit, silent)



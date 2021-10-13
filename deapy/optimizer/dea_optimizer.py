import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from core.abstract_dea_optimizer import AbstractDEAOptimizer
from utils.enums import Optimizer


class DEAOptimizer(AbstractDEAOptimizer):

    def __init__(self, optimizer: Optimizer = Optimizer.GLPK, time_limit: float = None, silent: bool = True):
        super(DEAOptimizer, self).__init__(optimizer, time_limit, silent)



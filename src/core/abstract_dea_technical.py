from abc import abstractmethod
from src.core.symbols import *
from src.core.abstract_dea import AbstractDEA
from nptyping import NDArray
import numpy as np


class AbstractDEATechnical(AbstractDEA):
    """
    An abstract class for technical dea models.
    """

    def __init__(self):
        self.eff = None
        self.slackX = None
        self.slackY = None

        super(AbstractDEA).__init__()

    @abstractmethod
    def efficiency(self) -> NDArray:
        """
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = DEARadial()

        >>> model.efficiency()

        >>> model.nobs()
        11

        Returns
        -------

        """

        return self.eff

    @abstractmethod
    def slacks(self, slack: Slack) -> NDArray:

        if slack == Slack.X:
            return self.slackX

        elif slack == Slack.Y:
            return self.slackY

        else:
            raise ValueError("`slack` must be Slack.X or Slack.Y")

    @abstractmethod
    def targets(self, target: Target) -> NDArray:

        if target == Target.X:
            return self.Xtarget

        elif target == Target.Y:
            return self.Ytarget

        else:
            raise ValueError("`target` must be Target.X or Target.Y")



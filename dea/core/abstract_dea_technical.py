from abc import abstractmethod
from dea.utils.symbols import *
from dea.core.abstract_dea import AbstractDEA
from nptyping import NDArray
import numpy as np


class AbstractDEATechnical(AbstractDEA):
    """
    An abstract class for technical dea models.
    """

    def __init__(self):

        self.slackX = None
        self.slackY = None
        self.lambdas = None
        self.Xtarget = None
        self.Ytarget = None

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

        >>> model.fit(X=X, Y=Y)

        >>> result = np.array([     \
                   [1.0000000000],  \
                   [0.6222896791],  \
                   [0.8198562444],  \
                   [1.0000000000],  \
                   [0.3103709311],  \
                   [0.5555555556],  \
                   [1.0000000000],  \
                   [0.7576690896],  \
                   [0.8201058201],  \
                   [0.4905660377],  \
                   [1.0000000000]   \
                ])

        >>> np.allclose(model.efficiency(), result, atol=1e-10)
        True
        >>> model.ndmu()
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



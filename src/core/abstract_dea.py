from abc import ABC, abstractmethod
from nptyping import NDArray
from typing import List
import numpy as np
import warnings


class AbstractDEA(ABC):
    """
    An abstract class for DEA Models.
    """

    def __init__(self):
        self.X = None
        self.Y = None
        self.n = None
        self.m = None
        self.s = None

        self.names = None
        self.slacksX = None
        self.slacksY = None
        self.lambdas = None
        self.Xtarget = None
        self.Ytarget = None

        super().__init__()

    @abstractmethod
    def dea(self):
        pass

    @abstractmethod
    def fit(self, X: NDArray, Y: NDArray) -> None:
        """

        Parameters
        ----------
        X -> inputs
        Y -> outputs

        Returns
        -------
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = DEARadial()

        >>> model.fit(X=X, Y=Y)

        >>> model.nobs()
        11
        >>> model.ninputs()
        2
        >>> model.noutputs()
        1

        """

        nx, self.m = X.shape
        ny, self.s = Y.shape

        if nx != ny:
            raise ValueError(f"number of rows in X and Y {nx, ny} are not equal")

        self.X = X
        self.Y = Y
        self.n = nx

    @abstractmethod
    def nobs(self) -> int:
        """
        Return number of observations in DEA model.
        Returns
        -------
        self.n
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = DEARadial()

        >>> model.fit(X=X, Y=Y)

        >>> model.nobs()
        11

        """
        return self.n

    @abstractmethod
    def ninputs(self) -> int:
        """
        Return number of inputs in DEA model.
        Returns
        -------
        self.m
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = DEARadial()

        >>> model.fit(X=X, Y=Y)

        >>> model.ninputs()
        2
        """
        return self.m

    @abstractmethod
    def noutputs(self) -> int:
        """
        Return number of outpus in DEA model.
        Returns
        -------
        self.s
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = DEARadial()

        >>> model.fit(X=X, Y=Y)

        >>> model.noutputs()
        1
        """

        return self.s

    @abstractmethod
    def dmunames(self) -> List[str]:
        """
        Return the names of decision making units (DMUS).
        Returns
        -------
        self.names
        >>> from src.technical.dea_radial import DEARadial

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15]])

        >>> X.shape
        (4, 2)
        >>> Y = np.array([[12], [14], [25], [26]])

        >>> Y.shape
        (4, 1)
        >>> dmunames = ["A", "B", "C", "D"]

        >>> model = DEARadial(names=dmunames)

        >>> model.fit(X=X, Y=Y)

        >>> model.dmunames()
        ['A', 'B', 'C', 'D']
        >>> dmunames = ["A", "B", "C"]

        >>> model = DEARadial(names=dmunames)

        >>> model.fit(X=X, Y=Y)

        >>> model.dmunames()
        ['A', 'B', 'C', '1']
        >>> dmunames = ["A", "B", "C", "D", "E"]

        >>> model = DEARadial(names=dmunames)

        >>> model.fit(X=X, Y=Y)

        >>> model.dmunames()
        ['A', 'B', 'C', 'D']
        """
        dmunames : List[str] = []

        if self.names:

            names_length = len(self.names)

            if names_length == self.nobs():
                dmunames = self.names

            elif names_length < self.nobs():
                warnings.warn("Length of names lower than number of observations")
                dmunames = self.names + [f"{i}" for i in range(1, self.nobs() - names_length + 1)]

            else:
                warnings.warn("Length of names greater than number of observations")
                dmunames = self.names[:self.nobs()]
        else:

            dmunames = [f"{i}" for i in range(self.nobs())]

        return dmunames

    @abstractmethod
    def pprint(self):
        pass


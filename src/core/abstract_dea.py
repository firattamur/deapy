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

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = RadialDEA()

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

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = RadialDEA()

        >>> model.fit(inputs=X, outputs=Y)

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

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = RadialDEA()

        >>> model.fit(inputs=X, outputs=Y)

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

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])

        >>> X.shape
        (11, 2)
        >>> Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

        >>> Y.shape
        (11, 1)
        >>> model = RadialDEA()

        >>> model.fit(inputs=X, outputs=Y)

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

        >>> X = np.array([[5, 13], [16, 12], [16, 26], [17, 15]])

        >>> X.shape
        (4, 2)
        >>> Y = np.array([[12], [14], [25], [26]])

        >>> Y.shape
        (4, 1)
        >>> dmunames = ["A", "B", "C", "D"]

        >>> model = RadialDEA()

        >>> model.fit(inputs=X, outputs=Y, names=dmunames)

        >>> model.dmunames()
        ["A", "B", "C", "D"]
        >>> dmunames = ["A", "B", "C"]

        >>> model.fit(inputs=X, outputs=Y, names=dmunames)

        >>> model.dmunames()
        ["A", "B", "C", "1"]
        >>> dmunames = ["A", "B", "C", "D", "E"]

        >>> model.fit(inputs=X, outputs=Y, names=dmunames)

        >>> model.dmunames()
        ["A", "B", "C", "D"]
        """
        retnames: list = []

        if self.names:

            nlen = len(self.names)

            if nlen == self.nobs():
                retnames = self.names

            elif nlen < self.nobs():
                warnings.warn("Length of names lower than number of observations")
                retnames = self.names + [f"{i}" for i in range(nlen + 1, self.nobs())]

            else:
                warnings.warn("Length of names greater than number of observations")
                retnames = self.names[:self.nobs()]
        else:

            retnames = [f"{i}" for i in range(self.nobs())]

        return retnames

    @abstractmethod
    def pprint(self):
        pass


import numpy as np
import pyomo.environ as pyo
from nptyping import NDArray
from src.utils.symbols import *
from src.optimizer.dea_optimizer import DEAOptimizer


class PyomoUtils:
    """
    Util functions for pyomo module.
    """

    @staticmethod
    def create_pyomo_model() -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        return model

    @staticmethod
    def solve_pyomo_model(model: pyo.AbstractModel, optimizer: DEAOptimizer):

        import os
        import sys

        path = os.path.dirname(os.path.abspath(__file__))

        solvername = 'glpk'
        solverpath_folder = path
        solverpath_exe = os.path.join(path, "glpsol")

        sys.path.append(solverpath_folder)

        opt = pyo.SolverFactory(solvername, executable=solverpath_exe)
        results = opt.solve(model, timelimit=optimizer.time_limit, tee=(not optimizer.silent))

        return results, model


class TechnicalDEAUtils:
    """
    Util functions for Technical DEA Models.
    """

    @staticmethod
    def getOnesAdditiveModelWeights(X: NDArray, Y: NDArray, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        # Standard Additive DEA Model
        if orient == Orient.Graph or orient == Orient.Input:
            rhoX = np.ones(X.shape)

        if orient == Orient.Graph or orient == Orient.Output:
            rhoY = np.ones(Y.shape)

        return rhoX, rhoY

    @staticmethod
    def getMIPAdditiveModelWeights(X: NDArray, Y: NDArray, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        # Measure of Inefficiency Proportions
        if orient == Orient.Graph or orient == Orient.Input:
            rhoX = 1 / X

        if orient == Orient.Graph or orient == Orient.Output:
            rhoY = 1 / Y

        return rhoX, rhoY

    @staticmethod
    def getNormalizedAdditiveModelWeights(X: NDArray, Y: NDArray, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        # Normalized weighted additive DEA model
        if orient == Orient.Graph or orient == Orient.Input:

            rhoX = np.zeros(X.shape)
            _, m = X.shape

            for i in range(m):
                rhoX[:, i] = 1.0 / np.std(X[:, i], ddof=1)

            rhoX[np.isinf(rhoX)] = 0

        if orient == Orient.Graph or orient == Orient.Output:

            rhoY = np.zeros(Y.shape)
            _, s = Y.shape

            for i in range(s):
                rhoY[:, i] = 1.0 / np.std(Y[:, i], ddof=1)

            rhoY[np.isinf(rhoY)] = 0

        return rhoX, rhoY

    @staticmethod
    def getRAMAdditiveModelWeights(X: NDArray, Y: NDArray, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        _, m = X.shape
        _, s = Y.shape

        normalization = 0

        if orient == Orient.Graph:
            normalization = m + s
        elif orient == Orient.Input:
            normalization = m
        else:
            normalization = s

        if orient == Orient.Graph or orient == Orient.Input:

            rhoX = np.zeros(X.shape)

            for i in range(m):
                rhoX[:, i] = 1 / (normalization * (np.max(X[:, i]), np.min(X[:, i])))

            rhoX[np.isinf(rhoX)] = 0

        if orient == Orient.Graph or orient == Orient.Output:

            rhoY = np.zeros(Y.shape)

            for i in range(s):
                rhoY[:, i] = 1 / (normalization * (np.max(Y[:, i]), np.min(Y[:, i])))

            rhoY[np.isinf(rhoY)] = 0

        return rhoX, rhoY

    @staticmethod
    def getBAMAdditiveModelWeights(X: NDArray, Y: NDArray, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        # Bounded Adjusted Measure

        _, m = X.shape
        _, s = Y.shape

        normalization = 0

        if orient == Orient.Graph:
            normalization = m + s
        elif orient == Orient.Input:
            normalization = m
        else:
            normalization = s

        if orient == Orient.Graph or orient == Orient.Input:

            rhoX = np.zeros(X.shape)
            minX = np.zeros((m, 1))

            for i in range(m):
                minX[i, 0] = np.min(X[:, i])
                rhoX[:, i] = 1 / (normalization * X[:, i] - minX[i, 0])

            rhoX[np.isinf(rhoX)] = 0

        if orient == Orient.Graph or orient == Orient.Output:

            rhoY = np.zeros(Y.shape)
            maxY = np.zeros((s, 1))

            for i in range(s):
                maxY[i, 0] = np.maximum(Y[:, i])
                rhoY = 1 / (normalization * (maxY[i, 0] - Y[:, i]))

            rhoY[np.isinf(rhoY)] = 0

        return rhoX, rhoY

    @staticmethod
    def getAdditiveModelWeights(X: NDArray, Y: NDArray, model: AdditiveModels, orient: Orient) -> (NDArray, NDArray):

        rhoX = None
        rhoY = None

        if model == AdditiveModels.Ones:
            rhoX, rhoY = TechnicalDEAUtils.getOnesAdditiveModelWeights(X=X, Y=Y, orient=orient)
        elif model == AdditiveModels.MIP:
            rhoX, rhoY = TechnicalDEAUtils.getMIPAdditiveModelWeights(X=X, Y=Y, orient=orient)
        elif model == AdditiveModels.NORM:
            rhoX, rhoY = TechnicalDEAUtils.getNormalizedAdditiveModelWeights(X=X, Y=Y, orient=orient)
        elif model == AdditiveModels.RAM:
            rhoX, rhoY = TechnicalDEAUtils.getRAMAdditiveModelWeights(X=X, Y=Y, orient=orient)
        elif model == AdditiveModels.BAM:
            rhoX, rhoY = TechnicalDEAUtils.getBAMAdditiveModelWeights(X=X, Y=Y, orient=orient)
        else:
            raise ValueError("Invalid Additive DEA Model!")

        if orient == Orient.Input:
            rhoY = np.ones(Y.shape)

        elif orient == Orient.Output:
            rhoX = np.ones(X.shape)

        return rhoX, rhoY

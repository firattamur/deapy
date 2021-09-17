import warnings
from tqdm import tqdm

import numpy as np
from scipy import sparse
import pyomo.environ as pyo
from prettytable import PrettyTable

from typing import List
from nptyping import NDArray

from src.utils.symbols import *
from src.optimizer.dea_optimizer import DEAOptimizer
from src.utils.utils import PyomoUtils, TechnicalDEAUtils
from src.core.abstract_dea_technical import AbstractDEATechnical


class DEAAdditive(AbstractDEATechnical):

    def __init__(self,
                 model: AdditiveModels = AdditiveModels.Ones,
                 orient: Orient = Orient.Graph,
                 rts: RTS = RTS.VRS,
                 slack: bool = True,
                 disposX: Dispos = Dispos.Strong,
                 disposY: Dispos = Dispos.Strong,
                 optimizer: DEAOptimizer = DEAOptimizer(optimizer=Optimizer.GLPK,
                                                        time_limit=None,
                                                        silent=False)):

        if disposX != Dispos.Strong and disposX != Dispos.Weak:
            raise ValueError("`disposX` must be Dispos.Strong or Dispos.Weak")

        if disposY != Dispos.Strong and disposX != Dispos.Weak:
            raise ValueError("`disposY` must be Dispos.Strong or Dispos.Weak")

        if orient != Orient.Input and orient != Orient.Output and orient != Orient.Graph:
            raise ValueError("`orient` must be Orient.Input or Orient.Output")

        if rts != RTS.CSR and rts != RTS.VRS:
            raise ValueError("`rts` must be RTS.CSR or RTS.VRS")

        self.model = model
        self.orient = orient
        self.rts = rts
        self.slack = slack
        self.disposX = disposX
        self.disposY = disposY
        self.optimizer = optimizer

        self.names = None
        self.slacksX = None
        self.slacksY = None
        self.lambdas = None
        self.Xtarget = None
        self.Ytarget = None

        super(AbstractDEATechnical).__init__()

    def dea(self,
            Xref: NDArray = None, Yref: NDArray = None,
            rhoX: NDArray = None, rhoY: NDArray = None):
        super(DEAAdditive, self).dea()

        # check if fit called before
        if self.X is None or self.Y is None:
            raise ValueError("need to call model.fit(X, Y) to set inputs and outputs before run model.dea()!")

        # check parameters
        if Xref is None:
            Xref = self.X

        if Yref is None:
            Yref = self.Y

        nrefx, mref = Xref.shape
        nrefy, sref = Yref.shape

        if nrefx != nrefy:
            raise ValueError(f"number of rows in Xref and Yref ({nrefx}, {nrefy}) are not equal!")

        if self.n_inp != mref:
            raise ValueError(f"number of cols in X and Xref ({self.n_inp}, {mref}) are not equal!")

        if self.n_out != sref:
            raise ValueError(f"number of cols in Y and Yref ({self.n_out}, {sref}) are not equal!")

        if rhoX is not None and rhoY is not None:
            self.model = AdditiveModels.Custom

        if self.model != AdditiveModels.Custom:
            rhoX, rhoY = TechnicalDEAUtils.getAdditiveModelWeights(self.X, self.Y, self.model, self.orient)

        if rhoX.shape != self.X.shape:
            raise ValueError(f"shape of rhoX and X {rhoX.shape, self.X.shape} are not equal!")

        if rhoY.shape != self.Y.shape:
            raise ValueError(f"shape of rhoY and Y {rhoY.shape, self.Y.shape} are not equal!")

        # parameters for additional condition in BAM model
        minXref = np.min(self.X, axis=0)
        maxYref = np.max(self.Y, axis=0)

        n = self.ndmu()
        nref = nrefx

        effi = np.zeros((n, 1))
        slackX = np.zeros((n, self.n_inp))
        slackY = np.zeros((n, self.n_out))
        lambdaeff = sparse.csr_matrix((n, nref)).toarray()

        for i in tqdm(range(n)):

            # value of inputs and outputs to evaluate
            x0 = self.X[i, :]
            y0 = self.Y[i, :]

            # value of weights to evaluate
            rhoX0 = rhoX[i, :]
            rhoY0 = rhoY[i, :]

            if self.disposX == Dispos.Weak:
                rhoX0 = np.zeros((1, n))

            if self.disposY == Dispos.Weak:
                rhoY0 = np.zeros((1, n))

            lp_model = PyomoUtils.create_pyomo_model()

            # create set of indices for inputs and outputs
            lp_model.dmu = pyo.RangeSet(1, self.n_dmu)
            lp_model.inp = pyo.RangeSet(1, self.n_inp)
            lp_model.out = pyo.RangeSet(1, self.n_out)

            # create variables for slackX, slackY and lambdas
            lp_model.sX = pyo.Var(lp_model.inp, within=pyo.NonNegativeReals)
            lp_model.sY = pyo.Var(lp_model.out, within=pyo.NonNegativeReals)
            lp_model.lambdas = pyo.Var(lp_model.dmu, within=pyo.NonNegativeReals)

            if self.orient == Orient.Graph:

                def obj_rule(lp_model):
                    return sum(rhoX0[j - 1] * lp_model.sX[j] for j in lp_model.inp) + \
                           sum(rhoY0[j - 1] * lp_model.sY[j] for j in lp_model.out)

                lp_model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

            elif self.orient == Orient.Input:

                def obj_rule(lp_model):
                    return sum(rhoX0[j - 1] * lp_model.sX[j] for j in lp_model.inp)

                lp_model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

            else:

                def obj_rule(lp_model):
                    return sum(rhoY0[j - 1] * lp_model.sY[j] for j in lp_model.out)

                lp_model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

            lp_model.constraints = pyo.ConstraintList()

            for j in range(self.n_inp):
                lhs = sum(Xref[t - 1, j] * lp_model.lambdas[t] for t in lp_model.dmu)
                rhs = x0[j] - lp_model.sX[j + 1]

                lp_model.constraints.add(expr=(lhs == rhs))

            for j in range(self.n_out):
                lhs = sum(Yref[t - 1, j] * lp_model.lambdas[t] for t in lp_model.dmu)
                rhs = y0[j] + lp_model.sY[j + 1]

                lp_model.constraints.add(expr=(lhs == rhs))

            # add return to scale constraints

            if self.rts == RTS.CSR:

                # add constants for BAM CSR model
                if self.model == AdditiveModels.BAM:
                    for j in range(1, self.n_inp):
                        lhs = sum(Xref[t - 1, j - 1] * lp_model.lambdas[t] for t in lp_model.dmu)
                        rhs = minXref[j - 1]

                        lp_model.constraints.add(expr=(lhs >= rhs))

                    for j in range(1, self.n_out):
                        lhs = sum(Yref[t - 1, j - 1] * lp_model.lambdas[t] for t in lp_model.dmu)
                        rhs = maxYref[j - 1]

                        lp_model.constraints.add(expr=(lhs <= rhs))

            else:
                lhs = sum(lp_model.lambdas[j] for j in lp_model.dmu)
                rhs = 1
                lp_model.constraints.add(expr=(lhs == 1))

            # fix values of slacks when weight are zero

            for j in range(self.n_inp):
                if rhoX0[j] == 0:
                    lp_model.sX[j + 1].fix(0)

            for j in range(self.n_out):
                if rhoY0[j] == 0:
                    lp_model.sY[j + 1].fix(0)

            # results, lp_model = PyomoUtils.solve_pyomo_model(model=lp_model, optimizer=self.optimizer)

            opt = pyo.SolverFactory("glpk")
            results = opt.solve(lp_model)

            if results.solver.termination_condition != pyo.TerminationCondition.optimal and results.solver.termination_condition != pyo.TerminationCondition.locallyOptimal:
                message = f"DMU {i} termination status: {results.solver.termination_condition}."
                warnings.warn(message)
                continue

            effi[i, :] = pyo.value(lp_model.obj)

            for j in range(self.ndmu()):
                lambdaeff[i, j] = pyo.value(lp_model.lambdas[j + 1])

            for j in range(self.ninp()):
                slackX[i, j] = pyo.value(lp_model.sX[j + 1])

            for j in range(self.nout()):
                slackY[i, j] = pyo.value(lp_model.sY[j + 1])

        # save results to model
        self.eff = effi

        self.slackX = slackX
        self.slackY = slackY
        self.lambdas = lambdaeff

        self.Xtarget = self.X - slackX
        self.Ytarget = self.Y - slackY

    def fit(self, X: NDArray, Y: NDArray) -> None:
        super(DEAAdditive, self).fit(X, Y)

    def dmunames(self) -> List[str]:
        return super(DEAAdditive, self).dmunames()

    def ndmu(self) -> int:
        return super(DEAAdditive, self).ndmu()

    def ninp(self) -> int:
        return super(DEAAdditive, self).ninp()

    def nout(self) -> int:
        return super(DEAAdditive, self).nout()

    def efficiency(self) -> NDArray:
        return super(DEAAdditive, self).efficiency()

    def slacks(self, slack: Slack) -> NDArray:
        return super(DEAAdditive, self).slacks(slack)

    def targets(self, target: Target) -> NDArray:
        return super(DEAAdditive, self).targets(target)

    def pprint(self):

        print("Additive DEA Model")
        print(f"DMUs = {self.ndmu()}", end="; ")
        print(f"Inputs = {self.ninp()}", end="; ")
        print(f"Ouputs = {self.nout()}")
        print(f"Orientation = {self.orient.value}", end="; ")
        print(f"Returns to scale = {self.rts.value}")

        if self.disposX == Dispos.Weak:
            print(f"Weak disposibility of inputs ")

        if self.disposY == Dispos.Weak:
            print(f"Weak disposibility of outputs")

        cols = ["", "efficiency"] + [f"Slack X{i}" for i in range(self.slackX.shape[1])] + [f"Slack Y{i}" for i in
                                                                                            range(self.slackY.shape[1])]
        t = PrettyTable(cols)

        for i in range(self.n_dmu):
            row = [i + 1, self.eff[i, 0]]

            for sx in range(self.slackX.shape[1]):
                row.append(self.slackX[i, sx])

            for sy in range(self.slackY.shape[1]):
                row.append(self.slackY[i, sy])

            t.add_row(row)

        print(t)


if __name__ == '__main__':
    X = np.array(
        [[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])
    Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

    additive_dea = DEAAdditive(model=AdditiveModels.MIP)
    additive_dea.fit(X, Y)

    additive_dea.pprint()

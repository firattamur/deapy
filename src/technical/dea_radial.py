import os
import sys
import warnings
import numpy as np
from typing import List
from scipy import sparse
import pyomo.environ as pyo
from nptyping import NDArray

# TODO: Handle module imports in another way if it is possible
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.symbols import *
from src.utils.utils import PyomoUtils
from src.technical.dea_additive import DEAAdditive
from src.optimizer.dea_optimizer import DEAOptimizer
from src.core.abstract_dea_technical import AbstractDEATechnical


class DEARadial(AbstractDEATechnical):

    def __init__(self,
                 names: List[str] = None,
                 orient: Orient = Orient.Input,
                 rts: RTS = RTS.CSR,
                 slack: bool = True,
                 disposX: Dispos = Dispos.Strong,
                 disposY: Dispos = Dispos.Strong,
                 optimizer: DEAOptimizer = DEAOptimizer(optimizer=Optimizer.GLPK,
                                                        time_limit=None,
                                                        silent=True)):

        if disposX != Dispos.Strong and disposX != Dispos.Weak:
            raise ValueError("`disposX` must be Dispos.Strong or Dispos.Weak")

        if disposY != Dispos.Strong and disposX != Dispos.Weak:
            raise ValueError("`disposY` must be Dispos.Strong or Dispos.Weak")

        if orient != Orient.Input and orient != Orient.Output:
            raise ValueError("`orient` must be Orient.Input or Orient.Output")

        if rts != RTS.CSR and rts != RTS.VRS:
            raise ValueError("`rts` must be RTS.CSR or RTS.VRS")

        self.orient = orient
        self.rts = rts
        self.slack = slack
        self.disposX = disposX
        self.disposY = disposY
        self.optimizer = optimizer
        self.names = names

        super(AbstractDEATechnical).__init__()

    # TODO: Keep dea function without any arguments.
    def dea(self):
        super(DEARadial, self).dea()

        # check if fit called before
        if self.X is None or self.Y is None:
            raise ValueError("need to call model.fit(X, Y) to set inputs and outputs before run model.dea()!")

        # check parameters
        if self.Xref is None:
            self.Xref = self.X

        if self.Yref is None:
            self.Yref = self.Y

        nrefx, mref = self.Xref.shape
        nrefy, sref = self.Yref.shape

        if nrefx != nrefy:
            raise ValueError(f"number of rows in Xref and Yref ({nrefx}, {nrefy}) are not equal!")

        if self.n_inp != mref:
            raise ValueError(f"number of cols in X and Xref ({self.n_inp}, {mref}) are not equal!")

        if self.n_out != sref:
            raise ValueError(f"number of cols in Y and Yref ({self.n_out}, {sref}) are not equal!")

        # Compute efficiency for each DMU
        n_dmu = self.ndmu()
        nref_dmu = nrefx

        effi = np.ones((n_dmu, 1))
        lambdaeff = sparse.csr_matrix((n_dmu, nref_dmu)).toarray()

        # for each dmu model lp problem and solve with pyomo
        for i in range(n_dmu):

            x0 = self.X[i, :]
            y0 = self.Y[i, :]

            # create and abstract pyomo model for lp
            lp_model = pyo.ConcreteModel()

            # create set of indices for dmus
            lp_model.n = pyo.RangeSet(1, nref_dmu)

            # create efficiency and lambda variables for each dmu
            lp_model.eff = pyo.Var(within=pyo.Reals)
            lp_model.lambdas = pyo.Var(lp_model.n, within=pyo.NonNegativeReals)

            if self.orient == Orient.Input:

                # input oriented
                lp_model.obj = pyo.Objective(expr=lp_model.eff, sense=pyo.minimize)

                # create list for constraints
                lp_model.constraints = pyo.ConstraintList()

                # inequality or equality restrictions based on disposability
                if self.disposX == Dispos.Strong:

                    for j in range(self.n_inp):
                        lhs = sum(lp_model.lambdas[i] * self.Xref[i - 1, j] for i in lp_model.n)
                        rhs = lp_model.eff * x0[j]
                        lp_model.constraints.add(expr=(lhs <= rhs))

                else:

                    for j in range(self.n_inp):
                        lhs = sum(lp_model.lambdas[i] * self.Xref[i - 1, j] for i in lp_model.n)
                        rhs = lp_model.eff * x0[j]
                        lp_model.constraints.add(expr=(lhs == rhs))

                if self.disposY == Dispos.Strong:

                    for j in range(self.n_out):
                        lhs = sum(lp_model.lambdas[i] * self.Yref[i - 1, j] for i in lp_model.n)
                        rhs = y0[j]
                        lp_model.constraints.add(expr=(lhs >= rhs))

                else:

                    for j in range(self.n_out):
                        lhs = sum(lp_model.lambdas[i] * self.Yref[i - 1, j] for i in lp_model.n)
                        rhs = y0[j]
                        lp_model.constraints.add(expr=(lhs == rhs))

            else:

                # output oriented
                lp_model.obj = pyo.Objective(expr=lp_model.eff, sense=pyo.maximize)

                # create list for constraints
                lp_model.constraints = pyo.ConstraintList()

                # inequality or equality restrictions based on disposability
                if self.disposX == Dispos.Strong:

                    for j in range(self.n_inp):
                        lhs = sum(lp_model.lambdas[i] * self.Xref[i - 1, j] for i in lp_model.n)
                        rhs = x0[j]
                        lp_model.constraints.add(expr=(lhs <= rhs))

                else:

                    for j in range(self.n_inp):
                        lhs = sum(lp_model.lambdas[i] * self.Xref[i - 1, j] for i in lp_model.n)
                        rhs = x0[j]
                        lp_model.constraints.add(expr=(lhs == rhs))

                if self.disposY == Dispos.Strong:

                    for j in range(self.n_out):
                        lhs = sum(lp_model.lambdas[i] * self.Yref[i - 1, j] for i in lp_model.n)
                        rhs = lp_model.eff * y0[j]
                        lp_model.constraints.add(expr=(lhs >= rhs))

                else:

                    for j in range(self.n_out):
                        lhs = sum(lp_model.lambdas[i] * self.Yref[i - 1, j] for i in lp_model.n)
                        rhs = lp_model.eff * y0[j]
                        lp_model.constraints.add(expr=(lhs == rhs))

            # add return to scale constraints
            if self.rts == RTS.CSR:
                pass  # no constraint to add for constant returns to scale
            else:
                lhs = sum(lp_model.lambdas[j] for j in lp_model.n)
                rhs = 1
                lp_model.constraints.add(expr=(lhs == 1))

            opt = pyo.SolverFactory("glpk")
            results = opt.solve(lp_model)

            if results.solver.termination_condition != pyo.TerminationCondition.optimal and \
                    results.solver.termination_condition != pyo.TerminationCondition.locallyOptimal:
                # warnings.warn(f"DMU {i} termination status: {results.solver.termination_condition}.")
                pass
            else:

                effi[i, :] = pyo.value(lp_model.eff)

                for j in range(self.ndmu()):
                    lambdaeff[i, j] = pyo.value(lp_model.lambdas[j + 1])

        # save results to model
        self.eff = effi
        self.lambdas = lambdaeff

        # get first-stage X and Y targets
        if self.orient == Orient.Input:
            self.Xtarget = self.X * effi
            self.Ytarget = self.Y

        else:
            self.Xtarget = self.X
            self.Ytarget = self.Y * effi

        # compute slacks
        if self.slack:

            # use additive model with radial efficient X and Y to get slacks
            if self.disposX == Dispos.Strong:
                rhoX = np.ones(self.X.shape)
            else:
                rhoX = np.zeros(self.X.shape)

            if self.disposY == Dispos.Strong:
                rhoY = np.ones(self.Y.shape)
            else:
                rhoY = np.zeros(self.Y.shape)

            slackModel = DEAAdditive(rts=self.rts, optimizer=self.optimizer, rhoX=rhoX, rhoY=rhoY)
            slackModel.fit(X=self.Xtarget, Y=self.Ytarget, Xref=self.Xref, Yref=self.Yref)

            self.slackX = slackModel.slacks(slack=Slack.X)
            self.slackY = slackModel.slacks(slack=Slack.Y)

            # get second-stage X and Y targets
            self.Xtarget = self.Xtarget - self.slackX
            self.Ytarget = self.Ytarget - self.slackY

        else:

            self.slackX = None
            self.slackY = None

    def fit(self, X: NDArray, Y: NDArray, Xref: NDArray = None, Yref: NDArray = None) -> None:
        super(DEARadial, self).fit(X, Y, Xref, Yref)

    def dmunames(self) -> List[str]:
        return super(DEARadial, self).dmunames()

    def ndmu(self) -> int:
        return super(DEARadial, self).ndmu()

    def ninp(self) -> int:
        return super(DEARadial, self).ninp()

    def nout(self) -> int:
        return super(DEARadial, self).nout()

    def efficiency(self) -> NDArray:
        return super(DEARadial, self).efficiency()

    def slacks(self, slack: Slack) -> NDArray:
        return super(DEARadial, self).slacks(slack)

    def targets(self, target: Target) -> NDArray:
        return super(DEARadial, self).targets(target)

    def pprint(self):

        print("Radial DEA Model")
        print(f"DMUs = {self.ndmu()}", end="; ")
        print(f"Inputs = {self.ninp()}", end="; ")
        print(f"Ouputs = {self.nout()}")
        print(f"Orientation = {self.orient.value}", end="; ")
        print(f"Returns to scale = {self.rts.value}")

        if self.disposX == Dispos.Weak:
            print(f"Weak disposibility of inputs ")

        if self.disposY == Dispos.Weak:
            print(f"Weak disposibility of outputs")

        from prettytable import PrettyTable
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

    input_crs_radial_dea = DEARadial(orient=Orient.Input, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
    input_crs_radial_dea.fit(X, Y)
    input_crs_radial_dea.pprint()

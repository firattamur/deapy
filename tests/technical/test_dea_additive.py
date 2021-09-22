import os
import sys
import unittest

import numpy as np

from typing import Dict
from nptyping import NDArray

# TODO: Handle module imports in another way if it is possible
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.symbols import *
from src.technical.dea_radial import DEAAdditive
from src.core.abstract_dea_technical import AbstractDEATechnical


class TestDEAAdditive(unittest.TestCase):
    """
    Tests for RadialDEA model.
    """

    # test data

    X = np.array(
        [[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])
    Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

    def test_inputs(self):
        self.assertEqual(self.X.shape, (11, 2))
        self.assertEqual(self.Y.shape, (11, 1))

    all_models: Dict[str, AbstractDEATechnical] = {}
    all_models_type: Dict[str, type] = {}
    all_models_ndmu: Dict[str, int] = {}
    all_models_ninp: Dict[str, int] = {}
    all_models_nout: Dict[str, int] = {}
    all_models_efficiency: Dict[str, NDArray] = {}
    all_models_slacks_X: Dict[str, NDArray] = {}
    all_models_slacks_Y: Dict[str, NDArray] = {}
    all_models_peersmatrix: Dict[str, NDArray] = {}
    all_models_weights: Dict[str, NDArray] = {}

    # ------------------------------------------------------------
    # TEST - Additive CRS
    # ------------------------------------------------------------

    additive_crs = DEAAdditive(rts=RTS.CSR, rhoX=np.ones(X.shape), rhoY=np.ones(Y.shape))
    additive_crs.fit(X=X, Y=Y)

    all_models["additive_crs"] = additive_crs
    all_models_type["additive_crs"] = DEAAdditive
    all_models_ndmu["additive_crs"] = 11
    all_models_ninp["additive_crs"] = 2
    all_models_nout["additive_crs"] = 1

    all_models_efficiency["additive_crs"] = np.array(
        [
            [0.0000000000],
            [10.769230770],
            [10.837837840],
            [0.0000000000],
            [22.153846150],
            [17.923076920],
            [0.0000000000],
            [12.076923080],
            [11.613793100],
            [35.000000000],
            [4.0000000000]
        ], dtype=np.float64)

    all_models_slacks_X["additive_crs"] = None
    all_models_slacks_Y["additive_crs"] = None

    all_models_peersmatrix["additive_crs"] = np.array([
        [1.0000000000, 0, 0, 0.0000000000, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 0.5384615385, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.1216216216, 0, 0, 0.9054054054, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 1.0000000000, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 0.3076923077, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 0.3461538462, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 0.0000000000, 0, 0, 1.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 1.1538461538, 0, 0, 0.0000000000, 0, 0, 0, 0,
         0.0000000000, 0, 0, 0.4689655172, 0, 0, 0.6965517241, 0, 0, 0, 0,
         0.0000000000, 0, 0, 1.0000000000, 0, 0, 0.0000000000, 0, 0, 0, 0,
         1.0000000000, 0, 0, 0.0000000000, 0, 0, 0.0000000000, 0, 0, 0, 0]
    ], dtype=np.float64)

    all_models_weights["additive_crs"] = None

    # ------------------------------------------------------------
    # TEST - Additive VRS
    # ------------------------------------------------------------

    additive_vrs = DEAAdditive(rts=RTS.VRS, rhoX=np.ones(X.shape), rhoY=np.ones(Y.shape))
    additive_vrs.fit(X=X, Y=Y)

    all_models["additive_vrs"] = additive_vrs
    all_models_type["additive_vrs"] = DEAAdditive
    all_models_ndmu["additive_vrs"] = 11
    all_models_ninp["additive_vrs"] = 2
    all_models_nout["additive_vrs"] = 1

    all_models_efficiency["additive_vrs"] = np.array(
        [

        ], dtype=np.float64)

    all_models_slacks_X["additive_vrs"] = None
    all_models_slacks_Y["additive_vrs"] = None

    all_models_peersmatrix["additive_vrs"] = np.array([
        [1.0000000000, 0, 0, 0, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.6666666667, 0, 0, 0, 0, 0, 0.33333333330, 0, 0, 0, 0],
        [0.0000000000, 0, 1, 0, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.0000000000, 0, 0, 1, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [1.0000000000, 0, 0, 0, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.0000000000, 0, 0, 0, 0, 1, 0.00000000000, 0, 0, 0, 0],
        [0.0000000000, 0, 0, 0, 0, 0, 1.00000000000, 0, 0, 0, 0],
        [0.0000000000, 0, 0, 0, 0, 0, 0.00000000000, 1, 0, 0, 0],
        [0.0000000000, 0, 0, 0, 0, 0, 0.00000000000, 0, 1, 0, 0],
        [0.0000000000, 0, 0, 1, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [1.0000000000, 0, 0, 0, 0, 0, 0.00000000000, 0, 0, 0, 0]
    ], dtype=np.float64)

    all_models_weights["additive_vrs"] = None

    # check weights equal to one equal to default model.

    # Default CRS
    additive_crs_default = DEAAdditive(rts=RTS.CSR)
    additive_crs_default.fit(X=X, Y=Y)

    # Default VRS
    additive_vrs_default = DEAAdditive(rts=RTS.VRS)
    additive_vrs_default.fit(X=X, Y=Y)

    def test_crs_default(self):
        self.assertTrue(np.allclose(self.additive_crs_default.efficiency(),
                                    self.additive_crs.efficiency(),
                                    atol=1e-12))

    def test_vrs_default(self):
        self.assertTrue(np.allclose(self.additive_vrs_default.efficiency(),
                                    self.additive_vrs.efficiency(),
                                    atol=1e-12))

    # check AdditiveModels.Ones equals to model with weights of ones
    additive_crs_ones = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.CSR)
    additive_crs_ones.fit(X=X, Y=Y)

    additive_crs_ones_target = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.CSR)
    additive_crs_ones_target.fit(X=additive_crs_ones.targets(target=Target.X),
                                 Y=additive_crs_ones.targets(target=Target.Y))

    def test_crs_ones(self):
        self.assertEqual(self.additive_crs_ones.model, AdditiveModels.Ones)
        self.assertTrue(np.allclose(self.additive_crs_ones.efficiency(),
                                    self.additive_crs_default.efficiency(),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_crs_ones_target.efficiency(),
                                    np.zeros((11, 1)),
                                    atol=1e-10))

    additive_vrs_ones = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.VRS)
    additive_vrs_ones.fit(X=X, Y=Y)

    additive_vrs_ones_target = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.VRS)
    additive_vrs_ones_target.fit(X=additive_vrs_ones.targets(target=Target.X),
                                 Y=additive_vrs_ones.targets(target=Target.Y))

    def test_vrs_ones(self):
        self.assertEqual(self.additive_vrs_ones.model, AdditiveModels.Ones)
        self.assertTrue(np.allclose(self.additive_vrs_ones.efficiency(),
                                    self.additive_vrs_default.efficiency(),
                                    atol=1e-12))
        self.assertTrue(np.allclose(self.additive_vrs_ones_target.efficiency(),
                                    np.zeros((11, 1)),
                                    atol=1e-12))

    # ------------------------------------------------------------
    # TEST - Additive MIP CRS
    # ------------------------------------------------------------

    additive_crs_mip = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.CSR)
    additive_crs_mip.fit(X=X, Y=Y)

    all_models["additive_crs_mip"] = additive_crs_mip
    all_models_type["additive_crs_mip"] = DEAAdditive
    all_models_ndmu["additive_crs_mip"] = 11
    all_models_ninp["additive_crs_mip"] = 2
    all_models_nout["additive_crs_mip"] = 1

    all_models_efficiency["additive_crs_mip"] = np.array([
        [0.0000000000],
        [0.7577160494],
        [0.4168399168],
        [0.0000000000],
        [2.2219512195],
        [1.1478260870],
        [0.0000000000],
        [0.4867909868],
        [0.4041184041],
        [1.0726153846],
        [0.2352941176]
    ], dtype=np.float64)

    all_models_slacks_X["additive_crs_mip"] = np.array([
        [0.000000000, 0.0000000000],
        [3.037037037, 6.8148148150],
        [0.000000000, 10.837837838],
        [0.000000000, 0.0000000000],
        [0.000000000, 0.0000000000],
        [8.000000000, 0.0000000000],
        [0.000000000, 0.0000000000],
        [7.384615385, 4.6923076920],
        [8.296296296, 2.5185185190],
        [0.000000000, 8.2000000000],
        [0.000000000, 4.0000000000]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_crs_mip"] = np.array([
        [0.000000000],
        [0.000000000],
        [0.000000000],
        [0.000000000],
        [17.77560976],
        [7.200000000],
        [0.000000000],
        [0.000000000],
        [0.000000000],
        [19.36000000],
        [0.000000000]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_crs_mip"] = None
    all_models_weights["additive_crs_mip"] = None

    # ------------------------------------------------------------
    # TEST - Additive MIP VRS
    # ------------------------------------------------------------

    additive_vrs_mip = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.VRS)
    additive_vrs_mip.fit(X=X, Y=Y)

    all_models["additive_vrs_mip"] = additive_vrs_mip
    all_models_type["additive_vrs_mip"] = DEAAdditive
    all_models_ndmu["additive_vrs_mip"] = 11
    all_models_ninp["additive_vrs_mip"] = 2
    all_models_nout["additive_vrs_mip"] = 1

    all_models_efficiency["additive_vrs_mip"] = np.array([
        [0.0000000000],
        [0.5075187970],
        [0.0000000000],
        [0.0000000000],
        [2.2039473684],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [1.0432234432],
        [0.2352941176]
    ], dtype=np.float64)

    all_models_slacks_X["additive_vrs_mip"] = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [17, 15],
        [0, 4]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_vrs_mip"] = np.array([
        [0.0000000000],
        [7.1052631580],
        [0.0000000000],
        [0.0000000000],
        [17.631578947],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [1.0000000000],
        [0.0000000000]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_vrs_mip"] = None
    all_models_weights["additive_vrs_mip"] = None

    # ------------------------------------------------------------
    # TEST - Additive Normalized CRS
    # ------------------------------------------------------------

    additive_crs_norm = DEAAdditive(model=AdditiveModels.NORM, rts=RTS.CSR)
    additive_crs_norm.fit(X=X, Y=Y)

    all_models["additive_crs_norm"] = additive_crs_norm
    all_models_type["additive_crs_norm"] = DEAAdditive
    all_models_ndmu["additive_crs_norm"] = 11
    all_models_ninp["additive_crs_norm"] = 2
    all_models_nout["additive_crs_norm"] = 1

    all_models_efficiency["additive_crs_norm"] = np.array([
        [0.0000000000],
        [1.3569256615],
        [1.7407259078],
        [0.0000000000],
        [2.6877810368],
        [1.6953153107],
        [0.0000000000],
        [1.6540884810],
        [1.1212042237],
        [4.0172855145],
        [0.6424624298]
    ], dtype=np.float64)

    all_models_slacks_X["additive_crs_norm"] = np.array([
        [0.0000000000, 0.0000000000],
        [3.0370370370, 6.8148148150],
        [0.0000000000, 10.837837838],
        [0.0000000000, 0.0000000000],
        [10.592592593, 11.037037037],
        [14.666666667, 2.6666666670],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 10.298429319],
        [8.2962962960, 2.5185185190],
        [17.925925926, 15.370370370],
        [0.0000000000, 4.0000000000]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_crs_norm"] = np.array([
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_crs_norm"] = None
    all_models_weights["additive_crs_norm"] = None

    # ------------------------------------------------------------
    # TEST - Additive Normalized VRS
    # ------------------------------------------------------------

    additive_vrs_norm = DEAAdditive(model=AdditiveModels.NORM, rts=RTS.VRS)
    additive_vrs_norm.fit(X=X, Y=Y)

    all_models["additive_vrs_norm"] = additive_vrs_norm
    all_models_type["additive_vrs_norm"] = DEAAdditive
    all_models_ndmu["additive_vrs_norm"] = 11
    all_models_ninp["additive_vrs_norm"] = 2
    all_models_nout["additive_vrs_norm"] = 1

    all_models_efficiency["additive_vrs_norm"] = np.array([
        [0.0000000000],
        [0.8049248943],
        [0.0000000000],
        [0.0000000000],
        [2.0149704908],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [3.9898943952],
        [0.6424624298]
    ], dtype=np.float64)

    all_models_slacks_X["additive_vrs_norm"] = np.array([
        [0, 0.00],
        [0, 0.65],
        [0, 0.00],
        [0, 0.00],
        [0, 2.95],
        [0, 0.00],
        [0, 0.00],
        [0, 0.00],
        [0, 0.00],
        [17, 15.00],
        [0, 4.00]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_vrs_norm"] = np.array([
        [0.00],
        [6.25],
        [0.00],
        [0.00],
        [13.75],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [1.00],
        [0.00]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_vrs_norm"] = None
    all_models_weights["additive_vrs_norm"] = None

    # ------------------------------------------------------------
    # TEST - Additive RAM CRS
    # ------------------------------------------------------------

    additive_vrs_norm = DEAAdditive(model=AdditiveModels.NORM, rts=RTS.VRS)
    additive_vrs_norm.fit(X=X, Y=Y)

    all_models["additive_vrs_norm"] = additive_vrs_norm
    all_models_type["additive_vrs_norm"] = DEAAdditive
    all_models_ndmu["additive_vrs_norm"] = 11
    all_models_ninp["additive_vrs_norm"] = 2
    all_models_nout["additive_vrs_norm"] = 1

    all_models_efficiency["additive_vrs_norm"] = np.array([
        [0.0000000000],
        [0.8049248943],
        [0.0000000000],
        [0.0000000000],
        [2.0149704908],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [3.9898943952],
        [0.6424624298]
    ], dtype=np.float64)

    all_models_slacks_X["additive_vrs_norm"] = np.array([
        [0, 0.00],
        [0, 0.65],
        [0, 0.00],
        [0, 0.00],
        [0, 2.95],
        [0, 0.00],
        [0, 0.00],
        [0, 0.00],
        [0, 0.00],
        [17, 15.00],
        [0, 4.00]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_vrs_norm"] = np.array([
        [0.00],
        [6.25],
        [0.00],
        [0.00],
        [13.75],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [1.00],
        [0.00]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_vrs_norm"] = None
    all_models_weights["additive_vrs_norm"] = None

    # ------------------------------------------------------------
    # TEST - Additive RAM VRS
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive BAM CRS
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive BAM VRS
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Custom
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Print
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Graph Orient
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Input Orient
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Output Orient
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Input Orient  - Weak Disposibility
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Additive Output Orient - Weak Disposibility
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - Vector and Matrix Input and Outputs
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TEST - RAM and BAM with Orientation
    # ------------------------------------------------------------

    def test_type(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_type[model] is None:
                continue

            with self.subTest(i=i):
                self.assertEqual(type(self.all_models[model]), self.all_models_type[model])

    def test_ndmu(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_ndmu[model] is None:
                continue

            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].ndmu(), self.all_models_ndmu[model])

    def test_ninp(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_ninp[model] is None:
                continue

            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].ninp(), self.all_models_ninp[model])

    def test_nout(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_nout[model] is None:
                continue

            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].nout(), self.all_models_nout[model])

    def test_efficiency(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_efficiency[model] is None:
                continue

            with self.subTest(i=i):
                self.assertTrue(np.allclose(self.all_models[model].efficiency(),
                                            self.all_models_efficiency[model],
                                            atol=1e-12))

    def test_slacks_X(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_slacks_X[model] is None:
                continue

            with self.subTest(i=i):
                self.assertTrue(
                    np.allclose(self.all_models[model].slacks(slack=Slack.X),
                                self.all_models_slacks_X[model],
                                atol=1e-12))

    def test_slacks_Y(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_slacks_Y[model] is None:
                continue

            with self.subTest(i=i):
                self.assertTrue(
                    np.allclose(self.all_models[model].slacks(slack=Slack.Y),
                                self.all_models_slacks_Y[model],
                                atol=1e-12))

    # TODO: Create Peers Abstract class and add peer matrix func to Abstract DEA Class
    # def test_peers_matrix(self):
    #
    #     for i, model in enumerate(self.all_models.keys()):
    #         with self.subTest(i=i):
    #             self.assertTrue(
    #                 np.allclose(self.all_models[model].peersmatrix(),
    #                             self.all_models_peersmatrix[model],
    #                             atol=1e-12))

    # def test_weights(self):
    #
    #     for i, model in enumerate(self.all_models.keys()):
    #         with self.subTest(i=i):
    #             self.assertTrue(
    #                 np.allclose(self.all_models[model].weights(),
    #                             self.all_models_weights[model],
    #                             atol=1e-12))


if __name__ == '__main__':
    unittest.main()

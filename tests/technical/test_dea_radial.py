import os
import sys
import unittest

import numpy as np

# TODO: Handle module imports in another way if it is possible
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.symbols import *
from src.technical.dea_radial import DEARadial


class TestRadialDea(unittest.TestCase):
    """
    Tests for RadialDEA model.
        Tests Overview:
            -> Models:
                -> Input Oriented CRS
                -> Output Oriented CRS
                -> Input Oriented VRS
                -> Output Oriented VRS
                -> No Slack
                -> Test Xref and Yref with Initial values
                -> Weak Disposability
                -> Strong Disposability
                -> DMU Names
                -> Progress Meter

            -> Functions
                -> test type
                -> test nobs
                -> test inputs
                -> test noutputs
                -> test efficiency
                -> test convert
                -> test slacks
                -> test peers matrix

    """

    # test data

    X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])
    Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

    def test_inputs(self):
        self.assertEqual(self.X.shape, (11, 2))
        self.assertEqual(self.Y.shape, (11, 1))

    all_models = {}
    all_models_type = {}
    all_models_nobs = {}
    all_models_inputs = {}
    all_models_noutputs = {}
    all_models_efficiency = {}
    all_models_slacks_X = {}
    all_models_slacks_Y = {}
    all_models_peersmatrix = {}

    # ------------------------------------------------------------
    # TEST - Input Oriented CRS
    # ------------------------------------------------------------

    # input oriented crs
    input_crs_radial_dea = DEARadial(orient=Orient.Input, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
    input_crs_radial_dea.fit(X, Y)
    input_crs_radial_dea.dea()

    all_models["input_csr"] = input_crs_radial_dea
    all_models_type["input_csr"] = DEARadial
    all_models_nobs["input_csr"] = 11
    all_models_inputs["input_csr"] = 2
    all_models_noutputs["input_csr"] = 1
    all_models_efficiency["input_csr"] = np.array([
                                                   [1.0000000000],
                                                   [0.6222896791],
                                                   [0.8198562444],
                                                   [1.0000000000],
                                                   [0.3103709311],
                                                   [0.5555555556],
                                                   [1.0000000000],
                                                   [0.7576690896],
                                                   [0.8201058201],
                                                   [0.4905660377],
                                                   [1.0000000000]
                                                ], dtype=np.float64)
    all_models_slacks_X["input_csr"] = np.array([
                                                [0.000000000,  0],
                                                [0.000000000,  0],
                                                [0.000000000,  0],
                                                [0.000000000,  0],
                                                [0.000000000,  0],
                                                [4.444444444,  0],
                                                [0.000000000,  0],
                                                [0.000000000,  0],
                                                [1.640211640,  0],
                                                [0.000000000,  0],
                                                [0.000000000,  4]
                                               ], dtype=np.float64)

    all_models_slacks_Y["input_csr"] = np.zeros((11, 1), dtype=np.float64)

    all_models_peersmatrix["input_csr"] = np.array([
                                                    [1.000000000,  0,  0, 0.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.4249783174,  0,  0, 0.10928013877,  0,  0,  0,  0],
                                                    [1.134321653,  0,  0, 0.4380053908,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 1.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.2573807721,  0,  0, 0.04844814534,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 0.33333333333,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 1.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 1.0348650979,  0,  0, 0.11457435013,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 1.14814814815,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.4905660377,  0,  0, 0.49056603774,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  1.000000000]
                                                    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Output Oriented CRS
    # ------------------------------------------------------------

    # output oriented crs
    output_crs_radial_dea = DEARadial(orient=Orient.Output, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
    output_crs_radial_dea.fit(X, Y)
    output_crs_radial_dea.dea()

    all_models["output_crs"] = output_crs_radial_dea
    all_models_type["output_crs"] = DEARadial
    all_models_nobs["output_crs"] = 11
    all_models_inputs["output_crs"] = 2
    all_models_noutputs["output_crs"] = 1
    all_models_efficiency["output_crs"] = np.array([
                                                    [1.0000000000],
                                                    [1.6069686410],
                                                    [1.2197260270],
                                                    [1.0000000000],
                                                    [3.2219512200],
                                                    [1.8000000000],
                                                    [1.0000000000],
                                                    [1.3198373980],
                                                    [1.2193548390],
                                                    [2.0384615380],
                                                    [1.0000000000]
                                                    ], dtype=np.float64)
    all_models_slacks_X["output_crs"] = np.array([
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [8.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [2.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  4]
                                                    ], dtype=np.float64)

    all_models_slacks_Y["output_crs"] = np.zeros((11, 1))

    all_models_peersmatrix["output_crs"] = np.array([
                                                    [1.000000000,  0,  0, 0.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.6829268293,  0,  0, 0.17560975610,  0,  0,  0,  0],
                                                    [1.383561644,  0,  0, 0.5342465753,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 1.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.8292682927,  0,  0, 0.15609756100,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 0.60000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 1.00000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 1.3658536585,  0,  0, 0.15121951220,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 0.0000000000,  0,  0, 1.40000000000,  0,  0,  0,  0],
                                                    [0.000000000,  0,  0, 1.0000000000,  0,  0, 1.00000000000,  0,  0,  0,  0],
                                                    [1.000000000,  0,  0, 0.0000000000,  0,  0, 0.00000000000,  0,  0,  0,  0]
                                                    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Input Oriented VRS
    # ------------------------------------------------------------

    # input oriented vrs
    input_vrs_radial_dea = DEARadial(orient=Orient.Input, rts=RTS.VRS, disposX=Dispos.Strong, disposY=Dispos.Strong)
    input_vrs_radial_dea.fit(X, Y)
    input_vrs_radial_dea.dea()

    all_models["input_vrs"] = input_vrs_radial_dea
    all_models_type["input_vrs"] = DEARadial
    all_models_nobs["input_vrs"] = 11
    all_models_inputs["input_vrs"] = 2
    all_models_noutputs["input_vrs"] = 1
    all_models_efficiency["input_vrs"] = np.array([
                                                    [1.0000000000],
                                                    [0.8699861687],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [0.7116402116],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [0.4931209269],
                                                    [1.0000000000]
                                                ], dtype=np.float64)
    all_models_slacks_X["input_vrs"] = np.array([
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000, 0],
                                                    [0.000000000,  4]
                                                ], dtype=np.float64)

    all_models_slacks_Y["input_vrs"] = np.array([
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [2.698412698],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000],
                                                    [0.000000000]
                                                ], dtype=np.float64)

    all_models_peersmatrix["input_vrs"] = np.array([
                                                    [1.00000000000,  0,  0,  0.0000000000,  0,  0.00000000000,  0.00000000000,  0,  0,  0,  0],
                                                    [0.52558782849,  0,  0,  0.0000000000,  0,  0.28423236510,  0.19017980640,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  1,  0.0000000000,  0,  0.00000000000,  0.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  1.0000000000,  0,  0.00000000000,  0.00000000000,  0,  0,  0,  0],
                                                    [0.56613756614,  0,  0,  0.0000000000,  0,  0.43386243390,  0.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  1.00000000000,  0.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0.00000000000,  1.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0.00000000000,  0.00000000000,  1,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0.00000000000,  0.00000000000,  0,  1,  0,  0],
                                                    [0.03711078928,  0,  0,  0.4433381608,  0,  0.00000000000,  0.51955105000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0.00000000000,  0.00000000000,  0,  0,  0,  1.000000000]
                                                ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Output Oriented VRS
    # ------------------------------------------------------------

    # output oriented vrs
    output_vrs_radial_dea = DEARadial(orient=Orient.Output, rts=RTS.VRS, disposX=Dispos.Strong, disposY=Dispos.Strong)
    output_vrs_radial_dea.fit(X, Y)
    output_vrs_radial_dea.dea()

    all_models["output_vrs"] = output_vrs_radial_dea
    all_models_type["output_vrs"] = DEARadial
    all_models_nobs["output_vrs"] = 11
    all_models_inputs["output_vrs"] = 2
    all_models_noutputs["output_vrs"] = 1
    all_models_efficiency["output_vrs"] = np.array([
                                                    [1.0000000000],
                                                    [1.5075187970],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [3.2039473680],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [1.0000000000],
                                                    [1.1923076920],
                                                    [1.0000000000]
                                                ], dtype=np.float64)
    all_models_slacks_X["output_vrs"] = np.array([
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [0.000000000,  0],
                                                    [5.000000000, 11],
                                                    [0.000000000,  4]
                                                ], dtype=np.float64)

    all_models_slacks_Y["output_vrs"] = np.zeros((11, 1))

    all_models_peersmatrix["output_vrs"] = np.array([
                                                    [1.00000000000,  0,  0,  0.0000000000,  0,  0,  0.00000000000,  0,  0,  0,  0],
                                                    [0.38157894737,  0,  0,  0.1710526316,  0,  0,  0.44736842110,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  1,  0.0000000000,  0,  0,  0.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  1.0000000000,  0,  0,  0.00000000000,  0,  0,  0,  0],
                                                    [0.03947368421,  0,  0,  0.7763157895,  0,  0,  0.18421052630,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  1,  0.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0,  1.00000000000,  0,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0,  0.00000000000,  1,  0,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0,  0.00000000000,  0,  1,  0,  0],
                                                    [0.00000000000,  0,  0,  0.0000000000,  0,  0,  0.00000000000,  0,  1,  0,  0],
                                                    [1.00000000000,  0,  0,  0.0000000000,  0,  0,  0.00000000000,  0,  0,  0,  0]
                                                ], dtype=np.float64)

    def test_type(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(type(self.all_models[model]), self.all_models_type[model])

    def test_nobs(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].nobs(), self.all_models_nobs[model])

    def test_ninputs(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].ninputs(), self.all_models_inputs[model])

    def test_noutputs(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].noutputs(), self.all_models_noutputs[model])

    def test_efficiency(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertTrue(np.allclose(self.all_models[model].efficiency(), self.all_models_efficiency[model],
                                        atol=1e-14))

    def test_slacks_X(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertTrue(np.allclose(self.all_models[model].slacks(slack=Slack.X), self.all_models_slacks_X[model],
                                        atol=1e-12))

    def test_slacks_Y(self):
        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertTrue(np.allclose(self.all_models[model].slacks(slack=Slack.Y), self.all_models_slacks_Y[model],
                                        atol=1e-12))

    # def test_peers_matrix(self):
    #     for model in self.all_models.keys():
    #         self.assertEqual(self.all_models[model].peermatrixs(), self.all_models_peersmatrix[model])


if __name__ == '__main__':
    unittest.main()

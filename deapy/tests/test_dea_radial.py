import unittest

from typing import Dict
from nptyping import NDArray

import numpy as np

from deapy.utils.enums import *
from deapy.technical.dea_radial import DEARadial
from deapy.core.abstract_dea_technical import AbstractDEATechnical


class TestDEARadial(unittest.TestCase):
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

    # ------------------------------------------------------------
    # TEST - Input Oriented CRS
    # ------------------------------------------------------------

    # input oriented crs
    input_crs_radial_dea = DEARadial(orient=Orient.Input, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
    input_crs_radial_dea.fit(X, Y)

    all_models["input_csr"] = input_crs_radial_dea
    all_models_type["input_csr"] = DEARadial
    all_models_ndmu["input_csr"] = 11
    all_models_ninp["input_csr"] = 2
    all_models_nout["input_csr"] = 1
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
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [4.444444444, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [1.640211640, 0],
        [0.000000000, 0],
        [0.000000000, 4]
    ], dtype=np.float64)

    all_models_slacks_Y["input_csr"] = np.zeros((11, 1), dtype=np.float64)

    all_models_peersmatrix["input_csr"] = np.array([
        [1.000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.4249783174, 0, 0, 0.10928013877, 0, 0, 0, 0],
        [1.134321653, 0, 0, 0.4380053908, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 1.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.2573807721, 0, 0, 0.04844814534, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 0.33333333333, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 1.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 1.0348650979, 0, 0, 0.11457435013, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 1.14814814815, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.4905660377, 0, 0, 0.49056603774, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 1.000000000]
    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Output Oriented CRS
    # ------------------------------------------------------------

    # output oriented crs
    output_csr_radial_dea = DEARadial(orient=Orient.Output, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
    output_csr_radial_dea.fit(X, Y)

    all_models["output_csr"] = output_csr_radial_dea
    all_models_type["output_csr"] = DEARadial
    all_models_ndmu["output_csr"] = 11
    all_models_ninp["output_csr"] = 2
    all_models_nout["output_csr"] = 1
    all_models_efficiency["output_csr"] = np.array([
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
    all_models_slacks_X["output_csr"] = np.array([
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [8.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [2.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 4]
    ], dtype=np.float64)

    all_models_slacks_Y["output_csr"] = np.zeros((11, 1))

    all_models_peersmatrix["output_csr"] = np.array([
        [1.000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.6829268293, 0, 0, 0.17560975610, 0, 0, 0, 0],
        [1.383561644, 0, 0, 0.5342465753, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 1.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.8292682927, 0, 0, 0.15609756100, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 0.60000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 1.00000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 1.3658536585, 0, 0, 0.15121951220, 0, 0, 0, 0],
        [0.000000000, 0, 0, 0.0000000000, 0, 0, 1.40000000000, 0, 0, 0, 0],
        [0.000000000, 0, 0, 1.0000000000, 0, 0, 1.00000000000, 0, 0, 0, 0],
        [1.000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0]
    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Input Oriented VRS
    # ------------------------------------------------------------

    # input oriented vrs
    input_vrs_radial_dea = DEARadial(orient=Orient.Input, rts=RTS.VRS, disposX=Dispos.Strong, disposY=Dispos.Strong)
    input_vrs_radial_dea.fit(X, Y)

    all_models["input_vrs"] = input_vrs_radial_dea
    all_models_type["input_vrs"] = DEARadial
    all_models_ndmu["input_vrs"] = 11
    all_models_ninp["input_vrs"] = 2
    all_models_nout["input_vrs"] = 1
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
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 4]
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
        [1.00000000000, 0, 0, 0.0000000000, 0, 0.00000000000, 0.00000000000, 0, 0, 0, 0],
        [0.52558782849, 0, 0, 0.0000000000, 0, 0.28423236510, 0.19017980640, 0, 0, 0, 0],
        [0.00000000000, 0, 1, 0.0000000000, 0, 0.00000000000, 0.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 1.0000000000, 0, 0.00000000000, 0.00000000000, 0, 0, 0, 0],
        [0.56613756614, 0, 0, 0.0000000000, 0, 0.43386243390, 0.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 1.00000000000, 0.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0.00000000000, 1.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0.00000000000, 0.00000000000, 1, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0.00000000000, 0.00000000000, 0, 1, 0, 0],
        [0.03711078928, 0, 0, 0.4433381608, 0, 0.00000000000, 0.51955105000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0.00000000000, 0.00000000000, 0, 0, 0, 1.000000000]
    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - Output Oriented VRS
    # ------------------------------------------------------------

    # output oriented vrs
    output_vrs_radial_dea = DEARadial(orient=Orient.Output, rts=RTS.VRS, disposX=Dispos.Strong, disposY=Dispos.Strong)
    output_vrs_radial_dea.fit(X, Y)

    all_models["output_vrs"] = output_vrs_radial_dea
    all_models_type["output_vrs"] = DEARadial
    all_models_ndmu["output_vrs"] = 11
    all_models_ninp["output_vrs"] = 2
    all_models_nout["output_vrs"] = 1
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
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [0.000000000, 0],
        [5.000000000, 11],
        [0.000000000, 4]
    ], dtype=np.float64)

    all_models_slacks_Y["output_vrs"] = np.zeros((11, 1))

    all_models_peersmatrix["output_vrs"] = np.array([
        [1.00000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.38157894737, 0, 0, 0.1710526316, 0, 0, 0.44736842110, 0, 0, 0, 0],
        [0.00000000000, 0, 1, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 1.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0],
        [0.03947368421, 0, 0, 0.7763157895, 0, 0, 0.18421052630, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 1, 0.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0, 1.00000000000, 0, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 1, 0, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 1, 0, 0],
        [0.00000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 1, 0, 0],
        [1.00000000000, 0, 0, 0.0000000000, 0, 0, 0.00000000000, 0, 0, 0, 0]
    ], dtype=np.float64)

    # ------------------------------------------------------------
    # TEST - No Slacks
    # ------------------------------------------------------------
    no_slack_dea = DEARadial(slack=False)
    no_slack_dea.fit(X=X, Y=Y)

    all_models["no_slack_dea"] = no_slack_dea
    all_models_type["no_slack_dea"] = DEARadial
    all_models_ndmu["no_slack_dea"] = 11
    all_models_ninp["no_slack_dea"] = 2
    all_models_nout["no_slack_dea"] = 1
    all_models_efficiency["no_slack_dea"] = np.array([
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

    all_models_slacks_X["no_slack_dea"] = None
    all_models_slacks_Y["no_slack_dea"] = None

    def test_type(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(type(self.all_models[model]), self.all_models_type[model])

    def test_ndmu(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].ndmu(), self.all_models_ndmu[model])

    def test_ninp(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].ninp(), self.all_models_ninp[model])

    def test_nout(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertEqual(self.all_models[model].nout(), self.all_models_nout[model])

    def test_efficiency(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):
                self.assertTrue(np.allclose(self.all_models[model].efficiency(), self.all_models_efficiency[model],
                                            atol=1e-14))

    def test_slacks_X(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):

                if self.all_models_slacks_X[model] is None:
                    self.assertEqual(self.all_models[model].slacks(slack=Slack.X), self.all_models_slacks_X[model])
                else:
                    self.assertTrue(
                        np.allclose(self.all_models[model].slacks(slack=Slack.X), self.all_models_slacks_X[model],
                                    atol=1e-12))

    def test_slacks_Y(self):

        for i, model in enumerate(self.all_models.keys()):
            with self.subTest(i=i):

                if self.all_models_slacks_Y[model] is None:
                    self.assertEqual(self.all_models[model].slacks(slack=Slack.Y), self.all_models_slacks_Y[model])
                else:
                    self.assertTrue(
                        np.allclose(self.all_models[model].slacks(slack=Slack.Y), self.all_models_slacks_Y[model],
                                    atol=1e-12))

    # TODO: Create Peers Abstract class and add peer matrix func to Abstract DEA Class
    # def test_peers_matrix(self):
    #     for model in self.all_models.keys():
    #         self.assertEqual(self.all_models[model].peersmatrix(), self.all_models_peersmatrix[model])

    # ------------------------------------------------------------
    # TEST - Test if one-by-one DEA using evaluation and reference sets match initial results
    # ------------------------------------------------------------

    def test_one_by_one(self):

        ndmu, ninp = self.X.shape
        _,    nout = self.Y.shape

        io_crs_dea_ref_efficiency = np.zeros((ndmu, 1))
        oo_crs_dea_ref_efficiency = np.zeros((ndmu, 1))

        io_vrs_dea_ref_efficiency = np.zeros((ndmu, 1))
        oo_vrs_dea_ref_efficiency = np.zeros((ndmu, 1))

        io_vrs_dea_ref_slackX = np.zeros(self.X.shape)
        io_vrs_dea_ref_slackY = np.zeros(self.X.shape)

        Xref = self.X.copy()
        Yref = self.Y.copy()

        for i in range(ndmu):
            Xeval = self.X[i, :].reshape(1, ninp)
            Yeval = self.Y[i, :].reshape(1, nout)

            io_crs_dea = DEARadial(orient=Orient.Input, rts=RTS.CSR)
            io_crs_dea.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            io_crs_dea_ref_efficiency[i] = io_crs_dea.efficiency()

            oo_crs_dea = DEARadial(orient=Orient.Output, rts=RTS.CSR)
            oo_crs_dea.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            oo_crs_dea_ref_efficiency[i] = oo_crs_dea.efficiency()

            io_vrs_dea = DEARadial(orient=Orient.Input, rts=RTS.VRS)
            io_vrs_dea.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            io_vrs_dea_ref_efficiency[i] = io_vrs_dea.efficiency()

            oo_vrs_dea = DEARadial(orient=Orient.Output, rts=RTS.VRS)
            oo_vrs_dea.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            oo_vrs_dea_ref_efficiency[i] = oo_vrs_dea.efficiency()

            io_vrs_dea_slackX = DEARadial(orient=Orient.Input, rts=RTS.VRS)
            io_vrs_dea_slackX.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            io_vrs_dea_ref_slackX[i] = io_vrs_dea_slackX.slacks(slack=Slack.X)

            io_vrs_dea_slackY = DEARadial(orient=Orient.Input, rts=RTS.VRS)
            io_vrs_dea_slackY.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
            io_vrs_dea_ref_slackY[i] = io_vrs_dea_slackY.slacks(slack=Slack.Y)

        self.assertTrue(np.allclose(io_crs_dea_ref_efficiency, self.all_models["input_csr"].efficiency(), atol=1e-12))
        self.assertTrue(np.allclose(oo_crs_dea_ref_efficiency, self.all_models["output_csr"].efficiency(), atol=1e-12))

        self.assertTrue(np.allclose(io_vrs_dea_ref_efficiency, self.all_models["input_vrs"].efficiency(), atol=1e-12))
        self.assertTrue(np.allclose(oo_vrs_dea_ref_efficiency, self.all_models["output_vrs"].efficiency(), atol=1e-12))

        self.assertTrue(np.allclose(io_vrs_dea_ref_slackX, self.all_models["input_vrs"].slacks(slack=Slack.X), atol=1e-12))
        self.assertTrue(np.allclose(io_vrs_dea_ref_slackY, self.all_models["input_vrs"].slacks(slack=Slack.Y), atol=1e-12))


if __name__ == '__main__':
    unittest.main()

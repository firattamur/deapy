import os
import sys
import unittest

import numpy as np

from typing import Dict
from nptyping import NDArray

import warnings

warnings.filterwarnings("ignore")

# TODO: Handle module imports in another way if it is possible
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from dea.utils.symbols import *
from dea.technical.dea_radial import DEAAdditive
from dea.core.abstract_dea_technical import AbstractDEATechnical


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

    additive_mip_crs = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.CSR)
    additive_mip_crs.fit(X=X, Y=Y)

    all_models["additive_mip_crs"] = additive_mip_crs
    all_models_type["additive_mip_crs"] = DEAAdditive
    all_models_ndmu["additive_mip_crs"] = 11
    all_models_ninp["additive_mip_crs"] = 2
    all_models_nout["additive_mip_crs"] = 1

    all_models_efficiency["additive_mip_crs"] = np.array([
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

    all_models_slacks_X["additive_mip_crs"] = np.array([
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

    all_models_slacks_Y["additive_mip_crs"] = np.array([
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

    all_models_peersmatrix["additive_mip_crs"] = None
    all_models_weights["additive_mip_crs"] = None

    # ------------------------------------------------------------
    # TEST - Additive MIP VRS
    # ------------------------------------------------------------

    additive_mip_vrs = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.VRS)
    additive_mip_vrs.fit(X=X, Y=Y)

    all_models["additive_mip_vrs"] = additive_mip_vrs
    all_models_type["additive_mip_vrs"] = DEAAdditive
    all_models_ndmu["additive_mip_vrs"] = 11
    all_models_ninp["additive_mip_vrs"] = 2
    all_models_nout["additive_mip_vrs"] = 1

    all_models_efficiency["additive_mip_vrs"] = np.array([
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

    all_models_slacks_X["additive_mip_vrs"] = np.array([
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

    all_models_slacks_Y["additive_mip_vrs"] = np.array([
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

    all_models_peersmatrix["additive_mip_vrs"] = None
    all_models_weights["additive_mip_vrs"] = None

    # ------------------------------------------------------------
    # TEST - Additive Normalized CRS
    # ------------------------------------------------------------

    additive_norm_crs = DEAAdditive(model=AdditiveModels.NORM, rts=RTS.CSR)
    additive_norm_crs.fit(X=X, Y=Y)

    all_models["additive_norm_crs"] = additive_norm_crs
    all_models_type["additive_norm_crs"] = DEAAdditive
    all_models_ndmu["additive_norm_crs"] = 11
    all_models_ninp["additive_norm_crs"] = 2
    all_models_nout["additive_norm_crs"] = 1

    all_models_efficiency["additive_norm_crs"] = np.array([
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

    all_models_slacks_X["additive_norm_crs"] = np.array([
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

    all_models_slacks_Y["additive_norm_crs"] = np.array([
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

    all_models_peersmatrix["additive_norm_crs"] = None
    all_models_weights["additive_norm_crs"] = None

    # ------------------------------------------------------------
    # TEST - Additive Normalized VRS
    # ------------------------------------------------------------

    additive_norm_vrs = DEAAdditive(model=AdditiveModels.NORM, rts=RTS.VRS)
    additive_norm_vrs.fit(X=X, Y=Y)

    all_models["additive_norm_vrs"] = additive_norm_vrs
    all_models_type["additive_norm_vrs"] = DEAAdditive
    all_models_ndmu["additive_norm_vrs"] = 11
    all_models_ninp["additive_norm_vrs"] = 2
    all_models_nout["additive_norm_vrs"] = 1

    all_models_efficiency["additive_norm_vrs"] = np.array([
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

    all_models_slacks_X["additive_norm_vrs"] = np.array([
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

    all_models_slacks_Y["additive_norm_vrs"] = np.array([
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

    all_models_peersmatrix["additive_norm_vrs"] = None
    all_models_weights["additive_norm_vrs"] = None

    # ------------------------------------------------------------
    # TEST - Additive RAM CRS
    # ------------------------------------------------------------

    additive_ram_crs = DEAAdditive(model=AdditiveModels.RAM, rts=RTS.CSR)
    additive_ram_crs.fit(X=X, Y=Y)

    all_models["additive_ram_crs"] = additive_ram_crs
    all_models_type["additive_ram_crs"] = DEAAdditive
    all_models_ndmu["additive_ram_crs"] = 11
    all_models_ninp["additive_ram_crs"] = 2
    all_models_nout["additive_ram_crs"] = 1

    all_models_efficiency["additive_ram_crs"] = np.array([
        [0.00000000000],
        [0.14094094094],
        [0.18063063063],
        [0.00000000000],
        [0.27937937938],
        [0.17657657658],
        [0.00000000000],
        [0.17164048866],
        [0.11671671672],
        [0.41766766767],
        [0.06666666667]
    ], dtype=np.float64)

    all_models_slacks_X["additive_ram_crs"] = np.array([
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

    all_models_slacks_Y["additive_ram_crs"] = np.array([
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

    all_models_peersmatrix["additive_ram_crs"] = None
    all_models_weights["additive_ram_crs"] = None

    # ------------------------------------------------------------
    # TEST - Additive RAM VRS
    # ------------------------------------------------------------

    additive_ram_vrs = DEAAdditive(model=AdditiveModels.RAM, rts=RTS.VRS)
    additive_ram_vrs.fit(X=X, Y=Y)

    all_models["additive_ram_vrs"] = additive_ram_vrs
    all_models_type["additive_ram_vrs"] = DEAAdditive
    all_models_ndmu["additive_ram_vrs"] = 11
    all_models_ninp["additive_ram_vrs"] = 2
    all_models_nout["additive_ram_vrs"] = 1

    all_models_efficiency["additive_ram_vrs"] = np.array([
        [0.00000000000],
        [0.10297482838],
        [0.00000000000],
        [0.00000000000],
        [0.25553012967],
        [0.00000000000],
        [0.00000000000],
        [0.00000000000],
        [0.00000000000],
        [0.41764590678],
        [0.06666666667]
    ], dtype=np.float64)

    all_models_slacks_X["additive_ram_vrs"] = np.array([
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

    all_models_slacks_Y["additive_ram_vrs"] = np.array([
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

    all_models_peersmatrix["additive_ram_vrs"] = None
    all_models_weights["additive_ram_vrs"] = None

    # ------------------------------------------------------------
    # TEST - Additive BAM CRS
    # ------------------------------------------------------------

    additive_bam_crs = DEAAdditive(model=AdditiveModels.BAM, rts=RTS.CSR)
    additive_bam_crs.fit(X=X, Y=Y)

    all_models["additive_bam_crs"] = additive_bam_crs
    all_models_type["additive_bam_crs"] = DEAAdditive
    all_models_ndmu["additive_bam_crs"] = 11
    all_models_ninp["additive_bam_crs"] = 2
    all_models_nout["additive_bam_crs"] = 1

    all_models_efficiency["additive_bam_crs"] = np.array([
        [0.0000000000],
        [0.4578892372],
        [0.3051750381],
        [0.0000000000],
        [0.6732181854],
        [0.3239568683],
        [0.0000000000],
        [0.5255235602],
        [0.1913580247],
        [0.6902867780],
        [0.1212121212]
    ], dtype=np.float64)

    all_models_slacks_X["additive_bam_crs"] = np.array([
        [0.0000000000, 0.0000000000],
        [4.1103448280, 6.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [13.000000000, 8.0000000000],
        [17.493670886, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 9.2251308900],
        [8.2962962960, 2.5185185190],
        [13.296296296, 13.518518519],
        [0.0000000000, 4.0000000000]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_bam_crs"] = np.array([
        [0.0000000000],
        [0.0000000000],
        [5.4931506849],
        [0.0000000000],
        [0.4520547945],
        [0.0000000000],
        [0.0000000000],
        [1.0000000000],
        [0.0000000000],
        [5.0000000000],
        [0.0000000000]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_bam_crs"] = None
    all_models_weights["additive_bam_crs"] = None

    # ------------------------------------------------------------
    # TEST - Additive BAM VRS
    # ------------------------------------------------------------

    additive_bam_vrs = DEAAdditive(model=AdditiveModels.BAM, rts=RTS.VRS)
    additive_bam_vrs.fit(X=X, Y=Y)

    all_models["additive_bam_vrs"] = additive_bam_vrs
    all_models_type["additive_bam_vrs"] = DEAAdditive
    all_models_ndmu["additive_bam_vrs"] = 11
    all_models_ninp["additive_bam_vrs"] = 2
    all_models_nout["additive_bam_vrs"] = 1

    all_models_efficiency["additive_bam_vrs"] = np.array([
        [0.0000000000],
        [0.1998936736],
        [0.0000000000],
        [0.0000000000],
        [0.4329710145],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.0000000000],
        [0.5713608345],
        [0.1212121212]
    ], dtype=np.float64)

    all_models_slacks_X["additive_bam_vrs"] = np.array([
        [0.0000000000, 0.0000000000],
        [6.5964912280, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [13.000000000, 1.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [0.0000000000, 0.0000000000],
        [5.0000000000, 11.000000000],
        [0.0000000000, 4.0000000000]
    ], dtype=np.float64)

    all_models_slacks_Y["additive_bam_vrs"] = np.array([
        [0],
        [0],
        [0],
        [0],
        [4],
        [0],
        [0],
        [0],
        [0],
        [5],
        [0]
    ], dtype=np.float64)

    all_models_peersmatrix["additive_bam_vrs"] = None
    all_models_weights["additive_bam_vrs"] = None

    # ------------------------------------------------------------
    # TEST - Additive Custom
    # ------------------------------------------------------------

    additive_custom_crs = DEAAdditive(rhoX=(1 / X), rhoY=(1 / Y), rts=RTS.CSR)
    additive_custom_crs.fit(X=X, Y=Y)

    def test_custom_csr(self):
        self.assertEqual(self.additive_custom_crs.model, AdditiveModels.Custom)
        self.assertTrue(np.allclose(self.additive_custom_crs.efficiency(),
                                    self.additive_mip_crs.efficiency(),
                                    atol=1e-10))

    additive_custom_vrs = DEAAdditive(rhoX=(1 / X), rhoY=(1 / Y), rts=RTS.VRS)
    additive_custom_vrs.fit(X=X, Y=Y)

    def test_custom_vsr(self):
        self.assertEqual(self.additive_custom_vrs.model, AdditiveModels.Custom)
        self.assertTrue(np.allclose(self.additive_custom_vrs.efficiency(),
                                    self.additive_mip_vrs.efficiency(),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive Weights Zero - Slacks Zero
    # ------------------------------------------------------------

    additive_weights_zero = DEAAdditive(rhoX=np.zeros(X.shape), rhoY=np.zeros(Y.shape))
    additive_weights_zero.fit(X=X, Y=Y)

    def test_weights_zero_slacks(self):
        self.assertTrue(np.allclose(self.additive_weights_zero.slacks(slack=Slack.X),
                                    np.zeros(self.X.shape),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_weights_zero.slacks(slack=Slack.Y),
                                    np.zeros(self.Y.shape),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive One-By-One DEA
    # ------------------------------------------------------------

    ndmu, ninp = X.shape
    _, nout = Y.shape

    additive_default_crs_ref_efficiency = np.zeros((ndmu, 1))
    additive_default_vrs_ref_efficiency = np.zeros((ndmu, 1))

    additive_custom_crs_ref_efficiency = np.zeros((ndmu, 1))
    additive_custom_vrs_ref_efficiency = np.zeros((ndmu, 1))

    additive_ones_crs_ref_efficiency = np.zeros((ndmu, 1))
    additive_ones_vrs_ref_efficiency = np.zeros((ndmu, 1))

    additive_mip_crs_ref_efficiency = np.zeros((ndmu, 1))
    additive_mip_vrs_ref_efficiency = np.zeros((ndmu, 1))

    Xref = X.copy()
    Yref = Y.copy()

    for i in range(ndmu):
        Xeval = X[i, :]
        Xeval = Xeval.reshape(1, ninp)

        Yeval = Y[i, :]
        Yeval = Yeval.reshape(1, nout)

        additive_default_crs_ref = DEAAdditive(rts=RTS.CSR)
        additive_default_crs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_default_crs_ref_efficiency[i] = additive_default_crs_ref.efficiency()

        additive_default_vrs_ref = DEAAdditive(rts=RTS.VRS)
        additive_default_vrs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_default_vrs_ref_efficiency[i] = additive_default_vrs_ref.efficiency()

        additive_custom_crs_ref = DEAAdditive(model=AdditiveModels.Custom, rts=RTS.CSR, rhoX=(1 / Xeval),
                                              rhoY=(1 / Yeval))
        additive_custom_crs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_custom_crs_ref_efficiency[i] = additive_custom_crs_ref.efficiency()

        additive_custom_vrs_ref = DEAAdditive(model=AdditiveModels.Custom, rts=RTS.VRS, rhoX=(1 / Xeval),
                                              rhoY=(1 / Yeval))
        additive_custom_vrs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_custom_vrs_ref_efficiency[i] = additive_custom_vrs_ref.efficiency()

        additive_ones_crs_ref = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.CSR)
        additive_ones_crs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_ones_crs_ref_efficiency[i] = additive_ones_crs_ref.efficiency()

        additive_ones_vrs_ref = DEAAdditive(model=AdditiveModels.Ones, rts=RTS.VRS)
        additive_ones_vrs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_ones_vrs_ref_efficiency[i] = additive_ones_vrs_ref.efficiency()

        additive_mip_crs_ref = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.CSR)
        additive_mip_crs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_mip_crs_ref_efficiency[i] = additive_mip_crs_ref.efficiency()

        additive_mip_vrs_ref = DEAAdditive(model=AdditiveModels.MIP, rts=RTS.VRS)
        additive_mip_vrs_ref.fit(X=Xeval, Y=Yeval, Xref=Xref, Yref=Yref)
        additive_mip_vrs_ref_efficiency[i] = additive_mip_vrs_ref.efficiency()

    def test_additive_default_ref(self):
        self.assertTrue(np.allclose(self.additive_crs_default.efficiency(),
                                    self.additive_default_crs_ref_efficiency,
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_vrs_default.efficiency(),
                                    self.additive_default_vrs_ref_efficiency,
                                    atol=1e-10))

    def test_additive_custom_ref(self):
        self.assertTrue(np.allclose(self.additive_custom_crs.efficiency(),
                                    self.additive_custom_crs_ref_efficiency,
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_custom_vrs.efficiency(),
                                    self.additive_custom_vrs_ref_efficiency,
                                    atol=1e-10))

    def test_additive_ones_ref(self):
        self.assertTrue(np.allclose(self.additive_crs_ones.efficiency(),
                                    self.additive_ones_crs_ref_efficiency,
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_vrs_ones.efficiency(),
                                    self.additive_ones_vrs_ref_efficiency,
                                    atol=1e-10))

    def test_additive_mip_ref(self):
        self.assertTrue(np.allclose(self.additive_mip_crs.efficiency(),
                                    self.additive_mip_crs_ref_efficiency,
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_mip_vrs.efficiency(),
                                    self.additive_mip_vrs_ref_efficiency,
                                    atol=1e-10))

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
                                            atol=1e-10))

    def test_slacks_X(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_slacks_X[model] is None:
                continue

            with self.subTest(i=i):
                self.assertTrue(
                    np.allclose(self.all_models[model].slacks(slack=Slack.X),
                                self.all_models_slacks_X[model],
                                atol=1e-10))

    def test_slacks_Y(self):

        for i, model in enumerate(self.all_models.keys()):

            if self.all_models_slacks_Y[model] is None:
                continue

            with self.subTest(i=i):
                self.assertTrue(
                    np.allclose(self.all_models[model].slacks(slack=Slack.Y),
                                self.all_models_slacks_Y[model],
                                atol=1e-10))

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

    # ------------------------------------------------------------
    # TEST - Additive Graph Orient
    # ------------------------------------------------------------

    X_O = np.array([[1], [2], [3], [2], [4]])
    Y_O = np.array([[2], [3], [4], [1], [3]])

    # test default orientation which is Graph
    additive_graph = DEAAdditive(orient=Orient.Graph)
    additive_graph.fit(X=X_O, Y=Y_O)

    additive_default = DEAAdditive()
    additive_default.fit(X=X_O, Y=Y_O)

    def test_default_orient(self):
        self.assertTrue(np.allclose(self.additive_graph.efficiency(),
                                    self.additive_default.efficiency(),
                                    atol=1e-10))

    # there test are different from original package because of multiple optimization solution
    # can be occurred from solver.
    def test_graph_orient(self):

        self.assertTrue(np.allclose(self.additive_graph.efficiency(),
                                    np.array([[0], [0], [0], [2.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_graph.slacks(slack=Slack.X),
                                    np.array([[0], [0], [0], [1.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_graph.slacks(slack=Slack.Y),
                                    np.array([[0], [0], [0], [1.0], [0.0]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive Input Orient
    # ------------------------------------------------------------

    additive_input = DEAAdditive(orient=Orient.Input)
    additive_input.fit(X=X_O, Y=Y_O)

    def test_input_orient(self):

        self.assertTrue(np.allclose(self.additive_input.efficiency(),
                                    np.array([[0], [0], [0], [1.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_input.slacks(slack=Slack.X),
                                    np.array([[0], [0], [0], [1.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_input.slacks(slack=Slack.Y),
                                    np.array([[0], [0], [0], [1.0], [0.0]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive Output Orient
    # ------------------------------------------------------------

    additive_output = DEAAdditive(orient=Orient.Output)
    additive_output.fit(X=X_O, Y=Y_O)

    def test_output_orient(self):

        self.assertTrue(np.allclose(self.additive_output.efficiency(),
                                    np.array([[0], [0], [0], [2.0], [1.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_output.slacks(slack=Slack.X),
                                    np.array([[0], [0], [0], [0.0], [1.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_output.slacks(slack=Slack.Y),
                                    np.array([[0], [0], [0], [2.0], [1.0]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive Input Orient  - Weak Disposibility
    # ------------------------------------------------------------

    additive_input_weak = DEAAdditive(orient=Orient.Input, disposY=Dispos.Weak)
    additive_input_weak.fit(X=X_O, Y=Y_O)

    def test_input_orient_weak(self):

        self.assertTrue(np.allclose(self.additive_input_weak.efficiency(),
                                    np.array([[0], [0], [0], [0.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_input_weak.slacks(slack=Slack.X),
                                    np.array([[0], [0], [0], [0.0], [2.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_input_weak.slacks(slack=Slack.Y),
                                    np.array([[0], [0], [0], [0.0], [0.0]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Additive Output Orient - Weak Disposibility
    # ------------------------------------------------------------

    additive_output_weak = DEAAdditive(orient=Orient.Output, disposX=Dispos.Weak)
    additive_output_weak.fit(X=X_O, Y=Y_O)

    def test_output_orient_weak(self):

        self.assertTrue(np.allclose(self.additive_output_weak.efficiency(),
                                    np.array([[0], [0], [0], [2.0], [0.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_output_weak.slacks(slack=Slack.X),
                                    np.array([[0], [0], [0], [0.0], [0.0]]),
                                    atol=1e-10))
        self.assertTrue(np.allclose(self.additive_output_weak.slacks(slack=Slack.Y),
                                    np.array([[0], [0], [0], [2.0], [0.0]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - Vector and Matrix Input and Outputs
    # ------------------------------------------------------------

    # input as matrix, output as vector
    X_MV = np.array([[2, 2], [1, 4], [4, 1], [4, 3], [5, 5], [6, 1], [2, 5], [1.6, 8]])
    Y_MV = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])

    additive_ones_matrix_vector = DEAAdditive(model=AdditiveModels.Ones)
    additive_ones_matrix_vector.fit(X_MV, Y_MV)

    def test_input_matrix_output_vector(self):

        self.assertTrue(np.allclose(self.additive_ones_matrix_vector.efficiency(),
                                    np.array([[0.0], [0.0], [0.0], [3.0], [6.0], [2.0], [3.0], [5.2]]),
                                    atol=1e-10))

    # input as vector, output as matrix
    X_VM = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    Y_VM = np.array([[7, 7], [4, 8], [8, 4], [3, 5], [3, 3], [8, 2], [6, 4], [1.5, 5]])

    additive_ones_vector_matrix = DEAAdditive(model=AdditiveModels.Ones)
    additive_ones_vector_matrix.fit(X_VM, Y_VM)

    def test_input_vector_output_matrix(self):

        self.assertTrue(np.allclose(self.additive_ones_vector_matrix.efficiency(),
                                    np.array([[0.0], [0.0], [0.0], [6.0], [8.0], [2.0], [4.0], [7.5]]),
                                    atol=1e-10))

    # input as vector, output as vector
    X_VV = np.array([[2], [4], [8], [12], [6], [14], [14], [9.412]])
    Y_VV = np.array([[1], [5], [8], [9], [3], [7], [9], [2.353]])

    additive_ones_vector_vector = DEAAdditive(model=AdditiveModels.Ones)
    additive_ones_vector_vector.fit(X_VV, Y_VV)

    def test_input_vector_output_vector(self):

        self.assertTrue(np.allclose(self.additive_ones_vector_vector.efficiency(),
                                    np.array([[0.0], [0.0], [0.0], [0.0], [4.0], [7.33333333], [2.0], [8.059]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - RAM with Different Orientations
    # ------------------------------------------------------------

    additive_ram_graph = DEAAdditive(model=AdditiveModels.RAM, orient=Orient.Graph)
    additive_ram_graph.fit(X=X_O, Y=Y_O)

    additive_ram_input = DEAAdditive(model=AdditiveModels.RAM, orient=Orient.Input)
    additive_ram_input.fit(X=X_O, Y=Y_O)

    additive_ram_output = DEAAdditive(model=AdditiveModels.RAM, orient=Orient.Output)
    additive_ram_output.fit(X=X_O, Y=Y_O)

    def test_ram_orientations(self):

        self.assertTrue(np.allclose(self.additive_ram_graph.efficiency(),
                                    np.array([[0], [0], [0], [1 / 3], [1 / 3]]),
                                    atol=1e-10))

        self.assertTrue(np.allclose(self.additive_ram_input.efficiency(),
                                    np.array([[0], [0], [0], [1 / 3], [2 / 3]]),
                                    atol=1e-10))

        self.assertTrue(np.allclose(self.additive_ram_output.efficiency(),
                                    np.array([[0], [0], [0], [2 / 3], [1 / 3]]),
                                    atol=1e-10))

    # ------------------------------------------------------------
    # TEST - BAM with Different Orientations
    # ------------------------------------------------------------

    additive_bam_graph = DEAAdditive(model=AdditiveModels.BAM, orient=Orient.Graph)
    additive_bam_graph.fit(X=X_O, Y=Y_O)

    additive_bam_input = DEAAdditive(model=AdditiveModels.BAM, orient=Orient.Input)
    additive_bam_input.fit(X=X_O, Y=Y_O)

    additive_bam_output = DEAAdditive(model=AdditiveModels.BAM, orient=Orient.Output)
    additive_bam_output.fit(X=X_O, Y=Y_O)

    def test_bam_orientations(self):

        self.assertTrue(np.allclose(self.additive_bam_graph.efficiency(),
                                    np.array([[0], [0], [0], [2 / 3], [2 / 3]]),
                                    atol=1e-10))

        self.assertTrue(np.allclose(self.additive_bam_input.efficiency(),
                                    np.array([[0], [0], [0], [1], [2 / 3]]),
                                    atol=1e-10))

        self.assertTrue(np.allclose(self.additive_bam_output.efficiency(),
                                    np.array([[0], [0], [0], [2 / 3], [1]]),
                                    atol=1e-10))


if __name__ == '__main__':
    unittest.main()

from enum import Enum


class AdditiveModels(Enum):
    Ones       = "Ones"         # Standard additive DEA model.
    MIP        = "MIP"          # Measure of Inefficiency Proportions. (Charnes et al., 1987; Cooper et al., 1999)
    Normalized = "Normalized"   # Normalized weighted additive DEA model. (Lovell and Pastor, 1995)
    RAM        = "RAM"          # Range Adjusted Measure. (Cooper et al., 1999)
    BAM        = "BAM"          # Bounded Adjusted Measure. (Cooper et al, 2011)
    Custom     = "Custom"       # User supplied weights.


class Optimizer(Enum):
    GLPK  = "glpk"
    IPOPT = "ipopt"


class Dispos(Enum):
    Strong = "Strong"
    Weak = "Weak"


class RTS(Enum):
    CSR = "CRS"
    VRS = "VRS"


class Orient(Enum):
    Input  = "Input"
    Output = "Output"
    Graph  = "Graph"


class Slack(Enum):
    X = "X"
    Y = "Y"


class Target(Enum):
    X = "X"
    Y = "Y"

from enum import Enum


class AllowedOperator(Enum):
    PLUS = "+"
    MUL = "*"
    DIV = "/"
    MINUS = "-"
    POW = "pow"
    FMOD = "fmod"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    EXP = "exp"
    LOG = "log"
    LOG_2 = "log2"
    LOG_10 = "log10"
    SQRT = "sqrt"
    SINH = "sinh"
    COSH = "cosh"
    TANH = "tanh"
    ASINH = "asinh"
    ACOSH = "acosh"
    ATANH = "atanh"
    ABS = "abs"
    FLOOR = "floor"
    CEIL = "ceil"
    ROUND = "round"
    SIGN = "sign"
    TGAMMA = "tgamma"
    LGAMMA = "lgamma"
    ERF = "erf"
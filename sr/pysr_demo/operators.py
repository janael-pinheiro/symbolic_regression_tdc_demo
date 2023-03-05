from enum import Enum

class AllowedBinaryOperator(Enum):
    PLUS = "plus"
    SUB = "sub"
    MULT = "mult"
    POW = "pow"
    DIV = "div"
    MOD = "mod"


class AllowUnaryOperator(Enum):
    COS = "cos"
    EXP = "exp"
    SIN = "sin"
    SQRT = "sqrt"
    SQUARE = "square"
    ABS = "abs"
    TAN = "tan"
    INV = "inv(x) = 1/x"
    CUBE = "cube"

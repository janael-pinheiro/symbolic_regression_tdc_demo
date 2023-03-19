from math import (acos, acosh, asin, asinh, atanh, ceil, cos, cosh, erf, exp,
                  floor, fmod, log, log2, log10, sin, sinh, sqrt, tan,
                  tanh, lgamma)
from types import MappingProxyType

from numpy import mod, sign, square
from sklearn.metrics import f1_score
from typing import Dict, Callable

def evaluate(observed, predicted):
    return f1_score(observed, predicted)


def eval_globals_factory() -> Dict[str, Dict[str, Callable]]:
    builtins = {
            "cos": cos,
            "sin": sin,
            "exp": exp,
            "sqrt": sqrt,
            "tan": tan,
            "square": square,
            "abs": abs,
            "Abs": abs,
            "mod": mod,
            "Mod": mod,
            "fmod": fmod,
            "round": round,
            "pow": pow,
            "sign": sign,
            "atanh": atanh,
            "erf": erf,
            "ceil": ceil,
            "floor": floor,
            "asin": asin,
            "acos": acos,
            "log": log,
            "log2": log2,
            "log10": log10,
            "sinh": sinh,
            "cosh": cosh,
            "tanh": tanh,
            "asinh": asinh,
            "acosh": acosh,
            "lgamma": lgamma}
    immutable_builtins = MappingProxyType(builtins)
    eval_globals = {"__builtins__": immutable_builtins}
    return eval_globals

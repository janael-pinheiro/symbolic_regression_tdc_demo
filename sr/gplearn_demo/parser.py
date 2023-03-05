from sr.equation_parser import EquationParser
from sympy import sympify


class GPlearnParser(EquationParser):
    def __init__(self):
        self.__converter = {
            'sub': lambda x, y : x - y,
            'div': lambda x, y : x/y,
            'mul': lambda x, y : x*y,
            'add': lambda x, y : x + y,
            'neg': lambda x    : -x,
            'pow': lambda x, y : x**y
            }

    def parse(self, equation: str) -> str:
        return sympify(equation, self.__converter)

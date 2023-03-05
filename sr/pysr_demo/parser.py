from sr.equation_parser import EquationParser


class PySREquationParser(EquationParser):
    def parse(self, equation: str) -> str:
        return str(equation)

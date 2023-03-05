from abc import ABC, abstractmethod

class EquationParser(ABC):
    @abstractmethod
    def parse(self, equation: str) -> str:
        ...

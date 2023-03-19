from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from numpy import array
from pandas import DataFrame

from sr.base_model import AbstractModel
from sr.equation_parser import EquationParser
from sr.gplearn_demo.operators import AllowedOperators
from sr.gplearn_demo.parser import GPlearnParser


@dataclass
class Model(AbstractModel):
    estimator: Union[SymbolicClassifier, SymbolicRegressor]
    equation_parser: EquationParser = GPlearnParser()

    def create(self) -> Any:
        ...

    def fit(self, features: DataFrame, target: DataFrame) -> Any:
        self.estimator.fit(features, target.values.ravel())

    def predict(self, equation: str, features: DataFrame) -> array:
        return self.estimator.predict(features)

    def best_equation(self) -> str:
        return self.equation_parser.parse(str(self.estimator._program))

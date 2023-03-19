from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from gplearn.genetic import SymbolicRegressor
from numpy import array
from pandas import DataFrame

from sr.base_model import AbstractModel
from sr.equation_parser import EquationParser
from sr.gplearn_demo.model import Model
from sr.gplearn_demo.operators import AllowedOperators
from sr.gplearn_demo.parser import GPlearnParser


@dataclass
class GPLearnRegressorModel(AbstractModel):
    feature_names: Union[List[str], None] = None
    random_state: int = 42
    equation_parser: EquationParser = GPlearnParser()
    n_generations: int = 100
    function_set: Tuple[str] = (operator.value for operator in AllowedOperators)
    parsimony_coefficient: float = 0.01

    def __post_init__(self) -> None:
        self.__base_estimator: Union[AbstractModel, None] = None

    def create(self) -> SymbolicRegressor:
        estimator = SymbolicRegressor(
        generations=self.n_generations,
            function_set=self.function_set,
            parsimony_coefficient=self.parsimony_coefficient,
            feature_names=self.feature_names,
            random_state=self.random_state)
        self.__base_estimator = Model(
            estimator=estimator,
            equation_parser=self.equation_parser)

    def fit(self, features: DataFrame, target: DataFrame) -> Any:
        self.__base_estimator.fit(features, target)

    def predict(self, equation: str, features: DataFrame) -> array:
        return self.__base_estimator.predict(
            equation=equation,
            features=features)

    def best_equation(self) -> str:
        return self.__base_estimator.best_equation()

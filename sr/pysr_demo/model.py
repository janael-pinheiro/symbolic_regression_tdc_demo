import logging
from dataclasses import dataclass, field
from os import cpu_count
from typing import Dict, List, Union

from numpy import array
from pandas import DataFrame
from pysr import PySRRegressor

from sr.base_model import AbstractModel, BaseModel
from sr.equation_parser import EquationParser
from sr.exceptions import NotCreatedModel
from sr.predictor import Predictor
from sr.pysr_demo.operators import AllowedBinaryOperator, AllowUnaryOperator
from sr.pysr_demo.parser import PySREquationParser


@dataclass
class PySRModel(AbstractModel):
    predictor: Predictor
    binary_operators: List[str] = field(default_factory=lambda: [bo.value for bo in AllowedBinaryOperator])
    unary_operators: List[str] = field(default_factory=lambda: [uo.value for uo in AllowUnaryOperator])
    equation_parser: EquationParser = PySREquationParser()
    extra_sympy_mappings: Union[Dict[str, str], None] = field(default_factory=lambda: {"inv": lambda x: 1 / x})
    n_iterations: int = 100
    n_populations: int = 15
    population_size: int = 33
    max_size:int = 20
    anneling: bool = False
    model: PySRRegressor = None
    model_selection: str = "accuracy"
    loss: str = "loss(x, y) = (x - y)^2"
    random_state: Union[int, None] = None
    deterministic: bool = False
    n_processes: int = field(default_factory=cpu_count)
    multithreading: bool = True

    def __post_init__(self):
        self.n_processes = self.__set_number_processes()
        self.deterministic = self.__set_deterministic_parameter()
        self.multithreading = self.__set_multithreading()
        self.random_state = self.__set_random_state()

    def create(self) -> PySRRegressor:
        self.model = PySRRegressor(
            niterations= self.n_iterations,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            extra_sympy_mappings=self.extra_sympy_mappings,
            loss=self.loss,
            model_selection=self.model_selection,
            populations=self.n_populations,
            population_size=self.population_size,
            maxsize=self.max_size,
            annealing=self.anneling,
            random_state=self.random_state,
            procs=self.n_processes,
            deterministic=self.deterministic,
            multithreading=self.multithreading
            )
        if self.random_state is not None:
            logging.info("\nUsing %s as random_state.\nUse random_state=%s in future experiments to ensure reproducibility.", self.random_state, self.random_state)
        return self.model

    def __set_number_processes(self) -> int:
        return 0 if self.random_state is not None or self.deterministic else cpu_count()

    def __set_deterministic_parameter(self) -> bool:
        return True if self.random_state == 0 or self.deterministic else bool(self.random_state)

    def __set_random_state(self) -> int:
        return 42 if self.random_state is None and self.deterministic else self.random_state

    def __set_multithreading(self) -> bool:
        return False if self.random_state == 0 else not bool(self.random_state)

    def fit(self, features: DataFrame, target: DataFrame) -> PySRRegressor:
        base_model = BaseModel(model=self.model)
        self.model = base_model.fit(features=features, target=target)
        return self.model

    def predict(self, equation: str, features: DataFrame) -> array:
        predictions = self.predictor.predict(
            equation=equation,
            features=features)
        return array(predictions)

    def best_equation(self) -> str:
        if self.model is None:
            raise NotCreatedModel("The model has not yet been created")
        return self.equation_parser.parse(str(self.model.sympy()))

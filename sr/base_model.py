from abc import ABC, abstractmethod
from pandas import DataFrame
from numpy import array
from typing import Any
from dataclasses import dataclass
import logging

from sr.exceptions import NotCreatedModel


class AbstractModel(ABC):
    @abstractmethod
    def create(self):
        ...

    @abstractmethod
    def fit(self, features: DataFrame, target: DataFrame) -> Any:
        ...

    @abstractmethod
    def predict(self, equation: str, features: DataFrame) -> array:
        ...

    @abstractmethod
    def best_equation(self) -> str:
        ...


@dataclass
class BaseModel(AbstractModel):
    model: AbstractModel

    def create(self):
        raise NotImplementedError()

    def fit(self, features: DataFrame, target: DataFrame) -> Any:
        if self.model is None:
            raise NotCreatedModel("The model has not yet been created")
        logging.info("Training model.")
        self.model.fit(features, target.values.ravel())
        return self.model

    def predict(self, equation: str, features: DataFrame) -> array:
        raise NotImplementedError()

    def best_equation(self) -> str:
        raise NotImplementedError()
    
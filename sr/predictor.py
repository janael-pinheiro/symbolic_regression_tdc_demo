from abc import ABC, abstractmethod
from typing import List

from numpy import array, where
from pandas import DataFrame
from dataclasses import dataclass

from sr.utils import eval_globals_factory
from sr.exceptions import InvalidData


class Predictor(ABC):
    @abstractmethod
    def predict(
        self,
        equation: str,
        features: DataFrame):
        ...


class BasePredictor(Predictor):
    def predict(self, equation: str, features: DataFrame) -> array:
        """
        The eval function is insecure.
        We restrict the namespace of eval with eval_globals.
        """
        eval_globals = eval_globals_factory()
        predictions = []
        equation = self.__adapt_equation(
            equation=equation,
            feature_names=features.columns)
        for _, row in features.iterrows():
            features_dict = {feature_name: row[feature_name] for feature_name in features.columns}
            eval_globals.update({"features_dict":features_dict})
            try:
                predicted = eval(equation, dict(eval_globals))
            except ValueError as exception:
                raise InvalidData from exception
            predictions.append(predicted)
        return predictions

    def __adapt_equation(
            self,
            equation: str,
            feature_names: List[str]) -> str:
        for feature_name in feature_names:
            equation = equation.replace(feature_name, f"features_dict['{feature_name}']")
        return equation


@dataclass
class BinaryClassifier(Predictor):
    base_predictor: Predictor

    def predict(self, equation: str, features: DataFrame) -> array:
        predictions = self.base_predictor.predict(equation=equation, features=features)
        binary_predictions = where(array(predictions) >= 0.5, 1, 0)
        return binary_predictions


@dataclass
class Regressor(Predictor):
    base_predictor: Predictor

    def predict(self, equation: str, features: DataFrame) -> array:
        return self.base_predictor.predict(equation=equation, features=features)

from abc import ABC, abstractmethod
from typing import List

from numpy import array, where
from pandas import DataFrame

from sr.utils import eval_globals_factory


class Predictor(ABC):
    @abstractmethod
    def predict(
        self,
        equation: str,
        features: DataFrame):
        ...


class BinaryClassifierPredictor(Predictor):
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
            eval_globals["__builtins__"].update({"features_dict":features_dict})
            predicted = eval(equation, eval_globals)
            predictions.append(predicted)
        binary_predictions = where(array(predictions) >= 0.5, 1, 0)
        return binary_predictions

    def __adapt_equation(
            self,
            equation: str,
            feature_names: List[str]) -> str:
        for feature_name in feature_names:
            equation = equation.replace(feature_name, f"features_dict['{feature_name}']")
        return equation

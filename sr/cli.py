from dataclasses import dataclass
from typing import Dict

import typer
from pandas import DataFrame

from sr.base_model import AbstractModel
from sr.dataset import Dataset
from sr.gplearn_demo.classifier_model import GPLearnClassifierModel
from sr.gplearn_demo.regressor_model import GPLearnRegressorModel
from sr.pysr_demo.model import PySRModel
from sr.turingbot_demo.model import TuringBotModel
from sr.predictor import BasePredictor, BinaryClassifier, Regressor
from enum import Enum


class ObjectiveEnum(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class Context:
    strategy: AbstractModel

    def execute(self, features: DataFrame, target: DataFrame) -> None:
        self.strategy.feature_names = features.columns
        self.strategy.create()
        self.strategy.fit(features=features, target=target)
        best_equation = self.strategy.best_equation()
        predictions = self.strategy.predict(equation=best_equation, features=features)
        print("Best equation:", best_equation, sep="\n")
        print("Predictions:", predictions)


app = typer.Typer()


def create_models(objective: ObjectiveEnum) -> Dict[str, AbstractModel]:
    base_predictor = BasePredictor()
    predictor =\
        BinaryClassifier(base_predictor=base_predictor)\
            if objective == ObjectiveEnum.CLASSIFICATION\
                else Regressor(base_predictor=base_predictor)
    return {
        "turingbot": TuringBotModel(timeout_seconds=10, predictor=predictor),
        "pysr": PySRModel(predictor=predictor),
        "gplearn": GPLearnClassifierModel() if objective == ObjectiveEnum.CLASSIFICATION else GPLearnRegressorModel()
    }


@app.command()
def execute(
    features_filepath: str = typer.Option(default="", help="Absolute or relative filepath"),
    target_filepath: str = typer.Option(default="", help="Absolute or relative filepath"),
    algorithm: str = typer.Option(default="", help="Valid algorithm options: pysr, turingbot, and gplearn"),
    objective: ObjectiveEnum = typer.Option(default=ObjectiveEnum.CLASSIFICATION, help="Objective: classification or regression")):

    models = create_models(objective=objective)
    model = models.get(algorithm, None)
    if model:
        features = Dataset.load(filepath=features_filepath)
        target = Dataset.load(filepath=target_filepath)
        context = Context(model)
        context.execute(features=features, target=target)
    else:
        print("Valid algorithm options: pysr, turingbot, and gplearn")


if __name__ == "__main__":
    app()

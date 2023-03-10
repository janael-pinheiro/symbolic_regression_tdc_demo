from dataclasses import dataclass
from typing import Dict

import typer
from pandas import DataFrame

from sr.base_model import AbstractModel
from sr.dataset import Dataset
from sr.gplearn_demo.classifier_model import GPLearnClassifierModel
from sr.pysr_demo.model import PySRModel
from sr.turingbot_demo.model import TuringBotModel
from sr.utils import evaluate


@dataclass
class Context:
    strategy: AbstractModel

    def execute(self, features: DataFrame, target: DataFrame) -> None:
        self.strategy.feature_names = features.columns
        self.strategy.create()
        self.strategy.fit(features=features, target=target)
        best_equation = self.strategy.best_equation
        predictions = self.strategy.predict(equation=best_equation, features=features)
        print("Best equation:", best_equation, sep="\n")
        print("Predictions:", predictions)
        print("Training set F1-score: ", evaluate(target, predictions))


app = typer.Typer()


def create_models() -> Dict[str, AbstractModel]:
    return {
        "turingbot": TuringBotModel(),
        "pysr": PySRModel(),
        "gplearn": GPLearnClassifierModel()
    }


@app.command()
def execute(
    features_filepath: str = typer.Option(default="", help="Absolute or relative filepath"),
    target_filepath: str = typer.Option(default="", help="Absolute or relative filepath"),
    algorithm: str = typer.Option(default="", help="Valid algorithm options: pysr, turingbot, and gplearn")):
    models = create_models()
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

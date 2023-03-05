from os.path import join

from sr.dataset import Dataset
from sr.turingbot_demo.model import TuringBotModel
from sr.turingbot_demo.operator_enum import AllowedOperator


def main() -> None:
    survived =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "survived.csv"))
    features =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "experiment_features.csv"))
    model = TuringBotModel(
        timeout_seconds=60,
        early_stop_condition=0.40,
        allowed_functions=[
            AllowedOperator.ABS,
            AllowedOperator.PLUS,
            AllowedOperator.MINUS,
            AllowedOperator.MUL,
            AllowedOperator.DIV,
            AllowedOperator.SIN,
            AllowedOperator.COS])
    model.create()
    model.fit(features, survived)
    best_equation = model.best_equation
    print("Best equation:", best_equation, sep="\n")
    predictions = model.predict(equation=best_equation, features=features)
    print("Predictions:", predictions, sep="\n")


if __name__ == "__main__":
    main()

from os.path import join

from sr.dataset import Dataset
from sr.pysr_demo.model import PySRModel
from sr.pysr_demo.operators import AllowedBinaryOperator, AllowUnaryOperator
from sr.pysr_demo.parser import PySREquationParser

def main() -> None:
    parser = PySREquationParser()
    model = PySRModel(
        binary_operators=[
            AllowedBinaryOperator.PLUS.value,
            AllowedBinaryOperator.DIV.value],
        unary_operators=[
            AllowUnaryOperator.COS.value,
            AllowUnaryOperator.EXP.value],
        equation_parser=parser,
        n_populations=30,
        population_size=36,
        anneling=True)
    model.create()
    survived =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "survived.csv"))
    features =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "train_experiment_features.csv"))
    _ = model.fit(features, survived)
    best_equation = model.best_equation
    print("Best equation:", best_equation, sep="\n")
    predictions = model.predict(equation=best_equation, features=features)
    print("Predictions: ", predictions, sep="\n")


if __name__ == "__main__":
    main()

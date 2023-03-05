from os.path import join

from sr.dataset import Dataset
from sr.pysr_demo.model import PySRModel


def main() -> None:
    model = PySRModel(
        n_iterations=100,
        n_populations=15,
        population_size=20,
        binary_operators=["+"],
        unary_operators=[
            "inv(x) = (1/x)",
            "per(x) = x * 0.323f0", # f0 -> identify float32 in Julia
            "quart(x) = x^4"],
        extra_sympy_mappings={
            "inv": lambda x: (1 / x),
            "per": lambda x: x * 0.323,
            "quart": lambda x: x**4})
    model.create()
    survived =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "survived.csv"))
    features =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "experiment_features.csv"))
    equations = model.fit(features, survived)
    best_equation = model.best_equation
    print("Equations:", equations, sep="\n")
    print("Best equation:", best_equation, sep="\n")
    predictions = model.predict(equation=best_equation, features=features)
    print("Predictions: ", predictions, sep="\n")

if __name__ == "__main__":
    main()

from os.path import join

from sr.dataset import Dataset
from sr.gplearn_demo.classifier_model import GPLearnClassifierModel
from sr.gplearn_demo.parser import GPlearnParser


def main() -> None:
    survived =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "survived.csv"))
    features =\
        Dataset.load(join("Datasets", "Titanic", "proceed", "experiment_features.csv"))
    parser = GPlearnParser()
    model = GPLearnClassifierModel(
        equation_parser=parser,
        feature_names=features.columns,
        n_generations=100)
    _ = model.create()
    model.fit(features, survived)
    best_equation = model.best_equation
    print("Best equation:", best_equation, sep="\n")
    predictions = model.predict(features=features, equation=best_equation)
    print("Predictions: ", predictions, sep="\n")


if __name__ == "__main__":
    main()

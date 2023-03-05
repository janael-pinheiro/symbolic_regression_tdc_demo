from gplearn.genetic import SymbolicClassifier


def test_when_valid_arguments_should_create_gplearn_model(gplearn_model):
    model = gplearn_model.create()
    assert isinstance(model, SymbolicClassifier)


def test_given_gplearn_model_when_valid_dataset_should_create_equation(
        gplearn_deterministic_model,
        titanic_features,
        titanic_target):
    model = gplearn_deterministic_model.create()
    assert isinstance(model, SymbolicClassifier)
    _ = gplearn_deterministic_model.fit(titanic_features, titanic_target)
    best_equation = gplearn_deterministic_model.best_equation
    assert str(best_equation) == "-2*Sex + sin(Pclass)"

from pysr import PySRRegressor


def test_when_valid_arguments_should_create_pysr_model(pysr_model):
    model = pysr_model.create()
    assert isinstance(model, PySRRegressor)


def test_when_valid_arguments_should_create_deterministic_pysr_model(deterministic_pysr_model):
    model = deterministic_pysr_model.create()
    assert isinstance(model, PySRRegressor)


def test_given_pysr_model_when_valid_dataset_should_create_equation(deterministic_pysr_model, titanic_features, titanic_target):
    model = deterministic_pysr_model.create()
    assert isinstance(model, PySRRegressor)
    _ = deterministic_pysr_model.fit(titanic_features, titanic_target)
    best_equation = deterministic_pysr_model.best_equation()
    assert best_equation == "exp(-Sex/cos(cos(exp(Parch) + exp(Pclass) + 1.1265569)))*cos(exp(1.88766044573514*Pclass))"

from pytest import fixture
from pathlib import PurePath

from sr.pysr_demo.model import PySRModel
from sr.pysr_demo.parser import PySREquationParser
from sr.pysr_demo.operators import AllowedBinaryOperator, AllowUnaryOperator
from sr.gplearn_demo.classifier_model import GPLearnClassifierModel
from sr.gplearn_demo.parser import GPlearnParser
from sr.dataset import Dataset
from sr.gplearn_demo.operators import AllowedOperators


@fixture(scope="function")
def pysr_parser():
    return PySREquationParser()


@fixture(scope="function")
def pysr_model(pysr_parser):
    model = PySRModel(
        binary_operators=[
            AllowedBinaryOperator.PLUS.value,
            AllowedBinaryOperator.DIV.value],
        unary_operators=[
            AllowUnaryOperator.COS.value,
            AllowUnaryOperator.EXP.value],
        equation_parser=pysr_parser,
        n_populations=50,
        population_size=76,
        anneling=True)
    return model


@fixture(scope="function")
def deterministic_pysr_model(pysr_parser):
    model = PySRModel(
        binary_operators=[
            AllowedBinaryOperator.PLUS.value,
            AllowedBinaryOperator.DIV.value],
        unary_operators=[
            AllowUnaryOperator.COS.value,
            AllowUnaryOperator.EXP.value],
        equation_parser=pysr_parser,
        n_populations=10,
        population_size=16,
        anneling=False,
        random_state=5)
    return model


@fixture(scope="function")
def gplearn_parser():
    return GPlearnParser()


@fixture(scope="function")
def gplearn_model(gplearn_parser, titanic_features):
    model = GPLearnClassifierModel(
        equation_parser=gplearn_parser,
        n_generations=100,
        feature_names=titanic_features.columns)
    return model


@fixture(scope="function")
def gplearn_deterministic_model(gplearn_parser, titanic_features):
    model = GPLearnClassifierModel(
        equation_parser=gplearn_parser,
        n_generations=100,
        random_state=42,
        feature_names=titanic_features.columns,
        function_set= (
            AllowedOperators.ADD.value,
            AllowedOperators.SUB.value,
            AllowedOperators.MUL.value,
            AllowedOperators.DIV.value,
            AllowedOperators.COS.value,
            AllowedOperators.SIN.value))
    return model


@fixture(scope="function")
def titanic_features():
    return Dataset.load(str(PurePath().joinpath("tests", "resources", "experiment_features.csv")))


@fixture(scope="function")
def titanic_target():
    return Dataset.load(str(PurePath().joinpath("tests", "resources", "survived.csv")))

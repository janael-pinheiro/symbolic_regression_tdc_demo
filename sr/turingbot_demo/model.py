import logging
import os
import signal
from dataclasses import dataclass, field
from os import system
from os.path import join
from shutil import which
from subprocess import Popen
from sys import float_info
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, List, Union

from numpy import array
from pandas import DataFrame
from tqdm import tqdm

from sr.base_model import AbstractModel
from sr.predictor import BinaryClassifierPredictor, Predictor
from sr.turingbot_demo.allow_target_delay_enum import AllowTargetDelayEnum
from sr.turingbot_demo.bound_search_mode_enum import BoundSearchModeEnum
from sr.turingbot_demo.force_all_variables_enum import ForceAllVariablesEnum
from sr.turingbot_demo.integer_constants_enum import IntegerConstantsEnum
from sr.turingbot_demo.operators import AllowedOperator
from sr.turingbot_demo.search_metric_enum import SearchMetricEnum
from sr.turingbot_demo.test_sample_enum import TestSampleEnum
from sr.turingbot_demo.train_test_split_enum import TrainTestSplitEnum
from sr.turingbot_demo.utils import ConfigurationFile


@dataclass
class TuringBotModel(AbstractModel):
    predictor: Predictor = BinaryClassifierPredictor()
    timeout_seconds: int = 300
    early_stop_condition: float = 0.01
    search_metric: SearchMetricEnum = SearchMetricEnum.RMSE
    train_test_split: TrainTestSplitEnum = TrainTestSplitEnum.NO_CROSS_VALIDATION
    test_sample: TestSampleEnum = TestSampleEnum.RANDOM
    train_test_seed: int = -1
    integer_constants: IntegerConstantsEnum = IntegerConstantsEnum.DISABLED
    bound_search_mode: BoundSearchModeEnum = BoundSearchModeEnum.DEACTIVATED
    maximum_formula_complexity: int = 60
    history_size: int = 20
    fscore_beta: int = 1
    allow_target_delay: AllowTargetDelayEnum = AllowTargetDelayEnum.YES
    force_all_variables: ForceAllVariablesEnum = ForceAllVariablesEnum.NO
    custom_formula: str = ""
    allowed_functions: List[AllowedOperator] = field(
        default_factory = lambda: list(AllowedOperator))

    def __post_init__(self):
        self.__configuration = ConfigurationFile(
            search_metric=self.search_metric,
            train_test_split=self.train_test_split,
            test_sample=self.test_sample,
            train_test_seed=self.train_test_seed,
            integer_constants=self.integer_constants,
            bound_search_mode=self.bound_search_mode,
            maximum_formula_complexity=self.maximum_formula_complexity,
            history_size=self.history_size,
            fscore_beta=self.fscore_beta,
            allow_target_delay=self.allow_target_delay,
            force_all_variables=self.force_all_variables,
            custom_formula=self.custom_formula,
            allowed_functions=self.allowed_functions)
        self.__equations_dict = {"Size": [], "Error": [], "Equation": []}
        self.__equations: Union[DataFrame, None] = None

    def create(self):
        """
        TuringBot does not allow instantiating an object like PySR and GPlearn.
        """

    def fit(self, features: DataFrame, target: DataFrame) -> Any:
        features["target"] = target.values
        dataset = features
        turingbot_executable_path = which("turingbot")
        with TemporaryDirectory() as temp_dir_name:
            output_filepath = join(temp_dir_name, "output.txt")
            filepath = join(temp_dir_name, "turingbot_dataset.csv")
            configuration_filepath = join(temp_dir_name, "configuration.conf")
            self.__configuration.generate(filepath=configuration_filepath)
            dataset.to_csv(
                filepath,
                index=False)
            command = f"exec {turingbot_executable_path} {filepath} {configuration_filepath} --outfile {output_filepath} 1>/dev/null 2>/dev/null"
            with Popen(command, shell=True) as process:
                sleep(10)
                for _ in tqdm(range(self.timeout_seconds)):
                    sleep(1)
                    self.__parse_equation_file(output_filepath)
                    system("clear")
                    print(self.__equations)
                    if self.__get_smaller_error() <= self.early_stop_condition:
                        break
                os.kill(process.pid+1, signal.SIGKILL)
    
    def __parse_equation_file(self, filepath: str):
        try:
            with open(filepath, mode="r", encoding="utf-8") as file_reader:
                    lines = file_reader.readlines()
                    for i in range(2, len(lines)):
                        line = lines[i].strip().split()
                        self.__equations_dict["Size"].append(line[0])
                        self.__equations_dict["Error"].append(line[1])
                        self.__equations_dict["Equation"].append(line[2])
                    self.__equations = DataFrame.from_dict(self.__equations_dict)
                    self.__equations_dict = {"Size": [], "Error": [], "Equation": []}
        except FileNotFoundError:
            logging.warning("Equations not yet available!")

    def predict(self, equation: str, features: DataFrame) -> array:
        predictions = self.predictor.predict(
            equation=equation,
            features=features)
        return array(predictions)

    def __get_smaller_error(self) -> float:
        try:
            return float(self.__equations.iloc[-1, -2])
        except AttributeError:
            logging.warning("Equations not yet available!")
            return float_info.max

    @property
    def best_equation(self) -> str:
        return self.__equations.iloc[-1, -1]

from pydantic.dataclasses import dataclass, Field
from typing import List

from sr.turingbot_demo.search_metric_enum import SearchMetricEnum
from sr.turingbot_demo.operator_enum import AllowedOperator
from sr.turingbot_demo.train_test_split_enum import TrainTestSplitEnum
from sr.turingbot_demo.test_sample_enum import TestSampleEnum
from sr.turingbot_demo.integer_constants_enum import IntegerConstantsEnum
from sr.turingbot_demo.bound_search_mode_enum import BoundSearchModeEnum
from sr.turingbot_demo.allow_target_delay_enum import AllowTargetDelayEnum
from sr.turingbot_demo.force_all_variables_enum import ForceAllVariablesEnum


@dataclass
class ConfigurationFile:
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
    allowed_functions: List[AllowedOperator] = Field(list(AllowedOperator))

    def generate(self, filepath: str):
        self.allowed_functions = " ".join([af.value for af in self.allowed_functions])
        with open(file=filepath, encoding="utf-8", mode="w") as file_writer:
            file_writer.write(f"search_metric = {self.search_metric.value}\n")
            file_writer.write(f"train_test_split = {self.train_test_split.value}\n")
            file_writer.write(f"test_sample = {self.test_sample.value}\n")
            file_writer.write(f"train_test_seed = {self.train_test_seed}\n")
            file_writer.write(f"integer_constants = {self.integer_constants.value}\n")
            file_writer.write(f"bound_search_mode = {self.bound_search_mode.value}\n")
            file_writer.write(f"maximum_formula_complexity = {self.maximum_formula_complexity}\n")
            file_writer.write(f"history_size = {self.history_size}\n")
            file_writer.write(f"fscore_beta = {self.fscore_beta}\n")
            file_writer.write(f"allow_target_delay = {self.allow_target_delay.value}\n")
            file_writer.write(f"force_all_variables = {self.force_all_variables.value}\n")
            file_writer.write(f"custom_formula = {self.custom_formula}\n")
            file_writer.write(f"allowed_functions = {self.allowed_functions}")

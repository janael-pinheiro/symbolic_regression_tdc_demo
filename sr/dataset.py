from pandas import DataFrame, read_csv


class Dataset:
    @classmethod
    def load(cls, filepath: str) -> DataFrame:
        return read_csv(filepath)

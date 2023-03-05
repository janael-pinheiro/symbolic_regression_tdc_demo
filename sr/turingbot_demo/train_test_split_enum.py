from enum import Enum


class TrainTestSplitEnum(Enum):
    NO_CROSS_VALIDATION = -1
    SPLIT_50 = 50
    SPLIT_60 = 60
    SPLIT_70 = 70
    SPLIT_75 = 75
    SPLIT_80 = 80
    SPLIT_100 = 100
    SPLIT_1000 = 1000
    SPLIT_10000 = 10000
    SPLIT_100000 = 100000

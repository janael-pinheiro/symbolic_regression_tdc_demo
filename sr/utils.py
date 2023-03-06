from typing import Dict, Any, List
from pandas import DataFrame
from sklearn.metrics import f1_score
from math import cos, exp, sin, sqrt, tan, fmod
from numpy import mod, square


def predict(equation: str, eval_globals: Dict[str, Any], features: DataFrame) -> List[float]:
    """
    The eval function is insecure.
    We restrict the namespace of eval with eval_globals.
    """
    predictions = []
    for feature_name in features.columns:
        equation = equation.replace(feature_name, f"features_dict['{feature_name}']")
    for _, row in features.iterrows():
        features_dict = {feature_name: row[feature_name] for feature_name in features.columns}
        eval_globals["__builtins__"].update({"features_dict":features_dict})
        predicted = eval(equation, eval_globals)
        predictions.append(predicted)
    return predictions


def evaluate(observed, predicted):
    return f1_score(observed, predicted)


def eval_globals_factory() -> Dict[str, str]:
    eval_globals = {"__builtins__": {
            "cos": cos,
            "sin": sin,
            "exp": exp,
            "sqrt": sqrt,
            "tan": tan,
            "square": square,
            "abs": abs,
            "Abs": abs,
            "mod": mod,
            "Mod": mod,
            "fmod": fmod,
            "round": round,
            "pow": pow,}}
    return eval_globals
from typing import Dict, Any, List
from pandas import DataFrame


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

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import *
import numpy as np
from typing import Any, Tuple

# open and read config file
config = load_config()

# define log_transformer
log_transformer = FunctionTransformer(np.log1p)

def build_pipeline() -> Tuple[Pipeline, Pipeline]:
    """ Build numerical pipeline. """
    # build log features pipeline (containing non-negative values)
    log_pipeline = Pipeline(steps=[
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])
    # build numerical features pipeline
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    return log_pipeline, num_pipeline

def build_preprocessor(log_pipe: Pipeline, num_pipe: Pipeline) -> Any:
    preprocessor = ColumnTransformer(transformers=[
        ('log', log_pipe, config['log_features']),
        ('num', num_pipe, config['num_features'])
    ])
    return preprocessor
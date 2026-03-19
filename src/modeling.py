from src.utils import *
from typing import Any, Dict, Tuple
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import BaggingClassifier as BGC, RandomForestClassifier as RFC, AdaBoostClassifier as ABC, GradientBoostingClassifier as GBC
from sklearn.metrics import recall_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# load config
config = load_config()

# build baseline model
def build_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Build and train a baseline DummyClassifier using a stratified strategy.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    recall_base: Any
        Recall score resulted from y_pred and y_train.
    """
    
    # create baseline object 
    base_model = DummyClassifier(strategy='stratified')

    # fit object to train data
    base_model.fit(X_train, y_train)

    # predict train data
    y_pred = base_model.predict(X_train)

    # calculate baseline recall
    recall_base = recall_score(y_train, y_pred)

    return recall_base

# build cv-train model
def build_cv_train(
        estimator: Any,
        preprocessor: Any,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame) -> Tuple[Any, Any, Any]:
    """
    Perform cross-validated model training with preprocessing and SMOTE pipeline.
    Evaluates the best model on training data and returns predictions + best model.

    Parameters
    ----------
    estimator : Any
        Machine learning estimator to train.
    preprocessor : Any
        Preprocessing transformer.
    params : dict
        Hyperparameter search space for RandomizedSearchCV.
    X_train : np.ndarray
        Training input features.
    y_train : np.ndarray
        Training labels.
    is_smote : bool
        Option whether using smote or non-smote in fitting the models

    Returns
    -------
    tuple
        best_model : Any  
            The chosen best model based on recall train and recall cv.
        recall_train : Any
            Recall scores based on fit train fold.
        recall_cv : Any
            Recall scores based on fit validation fold (test).
    """

    model = ImbPipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', estimator)
    ])

    # fit cv and train model
    cv_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        return_train_score=True,
        n_iter=config['n_iter'],
        scoring='recall',
        cv=config['n_cv'],
        n_jobs=config['n_jobs']
    )

    cv_model.fit(X_train, y_train)

    # evaluate models
    recall_train = cv_model.cv_results_['mean_train_score'].max()
    recall_cv= cv_model.cv_results_['mean_test_score'].max()

    # pick best model
    best_model = cv_model.best_estimator_

    # log best params
    mlflow.log_params(cv_model.best_params_)

    # log metrics
    mlflow.log_metric('best_recall_train', recall_train)
    mlflow.log_metric('best_recall_cv', recall_cv)

    # log model
    mlflow.sklearn.log_model(best_model, name='model')

    return best_model, recall_train, recall_cv

def build_test(
        estimator: Any,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Evaluate trained model on test data and log the recall score to MLflow.

    Parameters
    ----------
    estimator : Any
        Trained model or pipeline (already fitted) used for prediction.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Ground truth labels for test data.

    Returns
    -------
    y_pred: np.ndarray
        Series of label predicted data by estimator.
    recall_test: float
        Recall score computed on the test dataset.

    Notes
    -----
    - This function assumes that an MLflow run is already active.
    - It does not start or end an MLflow run.
    - Typically used after model selection via cross-validation.
    """

    y_pred = estimator.predict(X_test)

    # evaluate test score
    recall_test = recall_score(y_test, y_pred)

    # log results
    mlflow.log_metric('test_recall', recall_test)

    return y_pred, recall_test

def confusion(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[ConfusionMatrixDisplay, int, int, int, int]:
    """
    Generate and display a confusion matrix plot, and return the underlying values.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        Tuple[
            ConfusionMatrixDisplay,
            int, int, int, int
        ]:
            - ConfusionMatrixDisplay object
            - True Negative (TN)
            - False Positive (FP)
            - False Negative (FN)
            - True Positive (TP)
    """

    # define cm object
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    # plot the cm display
    display.plot()

    # unpack the each component value
    tn, fp, fn, tp = cm.ravel()

    # show the plot
    plt.show()

    return display, tn, fp, fn, tp

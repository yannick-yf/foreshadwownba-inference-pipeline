"""Pre Train Multiple models."""

import argparse
import joblib
from src.utils.logs import get_logger
from pathlib import Path
import yaml

import pandas as pd

from sklearn import datasets

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

logger = get_logger("MODEL_INFERENCE_REPORT_STEP", log_level="INFO")

def load_data(
    nba_games_inseason_dataset_path: str,
    nba_games_training_dataset_path: str,
    model_columns_path: str,
) -> tuple:
    """Load datasets and columns from configuration."""
    nba_games_training_dataset_final = pd.read_csv(nba_games_training_dataset_path)
    nba_games_inseason_dataset_final = pd.read_csv(nba_games_inseason_dataset_path)
    columns_selected = pd.read_csv(model_columns_path)
    columns_selected.columns = ["index", "columns_name"]
    column_to_select = columns_selected["columns_name"].values
    return nba_games_inseason_dataset_final, nba_games_training_dataset_final, column_to_select

def model_inference_report(
    inseason_dataset_file_path: str,
    training_dataset_file_path:str,
    model_columns: str,
    target_variable: str,
    group_cv_variable: str,
) -> None:
    """Run inference pipeline."""

    nba_games_inseason_dataset_final, nba_games_training_dataset_final, column_to_select = load_data(
        nba_games_training_dataset_path = training_dataset_file_path,
        nba_games_inseason_dataset_path = inseason_dataset_file_path,
        model_columns_path=model_columns,
    )

    list_columns_to_delete = [
        target_variable,
        group_cv_variable,
        "id_season",
        "tm",
        "opp",
    ]

    nba_games_inseason = nba_games_inseason_dataset_final.drop(
        list_columns_to_delete, 
        axis=1
        )
    nba_games_inseason = nba_games_inseason[
        column_to_select
        ]
    nba_games_training = nba_games_training_dataset_final.drop(
        list_columns_to_delete, 
        axis=1
        )
    nba_games_training = nba_games_training[column_to_select][
            nba_games_training['game_nb'] <= int(nba_games_inseason.game_nb.max())
            ]

    # Data Stability Report
    data_stability= TestSuite(tests=[
        DataStabilityTestPreset(),
    ])
    data_stability.run(
        current_data=nba_games_inseason, 
        reference_data=nba_games_training, 
        column_mapping=None
        )
    data_stability.save_html(
        "./data/output/data_stability_report.html"
        )

    # Data Report Report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(
        current_data=nba_games_inseason, 
        reference_data=nba_games_training, 
        column_mapping=None
        )
    data_drift_report.save_html(
        "./data/output/data_drift_report.html"
        )

    logger.info("Inference Step Done")


def get_args():
    """
    Parse command line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    _dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params-file",
        type=Path,
        default="params.yaml",
    )

    args, _ = parser.parse_known_args()
    params = yaml.safe_load(args.params_file.open())

    get_dataset_params = params["get_dataset"]
    get_models_params = params["get_models"]
    inference_params = params["inference"]

    parser.add_argument(
        "--inseason-dataset-file-path",
        dest="inseason_dataset_file_path",
        type=str,
        default=get_dataset_params["inseason_dataset"],
    )

    parser.add_argument(
        "--training-dataset-file-path",
        dest="training_dataset_file_path",
        type=str,
        default=get_dataset_params["training_dataset"],
    )

    parser.add_argument(
        "--model-columns",
        dest="model_columns",
        type=str,
        default=get_models_params["model_columns"],
    )

    parser.add_argument(
        "--target-variable",
        dest="target_variable",
        type=str,
        default=inference_params["target_variable"],
    )

    parser.add_argument(
        "--group-cv-variable",
        dest="group_cv_variable",
        type=str,
        default=inference_params["group_cv_variable"],
    )

    args = parser.parse_args()

    return args


def main():
    """Run the Pre Train Multiple Models Pipeline."""
    args = get_args()

    model_inference_report(
        inseason_dataset_file_path=args.inseason_dataset_file_path,
        training_dataset_file_path=args.training_dataset_file_path,
        model_columns=args.model_columns,
        target_variable=args.target_variable,
        group_cv_variable=args.group_cv_variable
    )

if __name__ == "__main__":
    main()

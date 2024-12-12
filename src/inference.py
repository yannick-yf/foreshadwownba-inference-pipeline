"""Pre Train Multiple models."""

import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from src.utils.logs import get_logger
from pathlib import Path

logger = get_logger("INFERENCE_STEP", log_level="INFO")


def rename_opponent_columns(training_df: pd.DataFrame) -> pd.DataFrame:
    """Rename opponent columns."""
    training_df.columns = training_df.columns.str.replace("_y", "_opp")
    training_df.columns = training_df.columns.str.replace("_x", "")
    return training_df


def load_data(
    nba_games_inseason_dataset_path: str,
    model_columns_path: str,
) -> tuple:
    """Load datasets and columns from configuration."""
    nba_games_inseason_dataset_final = pd.read_csv(nba_games_inseason_dataset_path)
    columns_selected = pd.read_csv(model_columns_path)
    columns_selected.columns = ["index", "columns_name"]
    column_to_select = columns_selected["columns_name"].values
    return nba_games_inseason_dataset_final, column_to_select


def prepare_data(
    nba_games_inseason_dataset_final: pd.DataFrame,
    column_to_select: list,
    target_column: str,
    group_cv_variable: str,
) -> tuple:
    """Prepare input features and target variables."""

    list_columns_to_delete = [
        target_column,
        group_cv_variable,
        "id_season",
        "tm",
        "opp",
    ]

    y_inseason = nba_games_inseason_dataset_final.loc[:, target_column].values
    x_inseason = nba_games_inseason_dataset_final.drop(list_columns_to_delete, axis=1)
    x_inseason = x_inseason[column_to_select].values

    return x_inseason, y_inseason


def predict_and_create_dataframe(
    model,
    x_inseason: np.ndarray,
    nba_games_inseason_dataset_final: pd.DataFrame,
    target_column: str,
    group_cv_variable: str,
) -> pd.DataFrame:
    """Perform predictions and create the prediction DataFrame."""

    prediction_value = model.predict(x_inseason)
    prediction_proba = model.predict_proba(x_inseason)
    prediction_proba_df = pd.DataFrame(
        prediction_proba, columns=["prediction_proba_df_0", "prediction_proba_df_1"]
    )

    nba_games_inseasonn_w_pred = nba_games_inseason_dataset_final[
        [target_column, group_cv_variable, "id_season", "game_date", "tm", "opp"]
    ].copy()

    nba_games_inseasonn_w_pred["prediction_proba_df_0"] = prediction_proba_df[
        "prediction_proba_df_0"
    ]
    nba_games_inseasonn_w_pred["prediction_proba_df_1"] = prediction_proba_df[
        "prediction_proba_df_1"
    ]
    nba_games_inseasonn_w_pred["prediction_value"] = prediction_value

    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred.rename(
        columns={
            "prediction_proba_df_0": "prediction_proba_df_loose",
            "prediction_proba_df_1": "prediction_proba_df_win",
        }
    )
    return nba_games_inseasonn_w_pred


def add_opponent_features(nba_games_inseasonn_w_pred: pd.DataFrame) -> pd.DataFrame:
    """Add opponent features to the prediction DataFrame."""
    opponent_features = [
        "id",
        "tm",
        "opp",
        "prediction_proba_df_loose",
        "prediction_proba_df_win",
    ]
    nba_games_inseasonn_w_pred_opp = nba_games_inseasonn_w_pred[opponent_features]

    nba_games_inseasonn_w_pred = pd.merge(
        nba_games_inseasonn_w_pred,
        nba_games_inseasonn_w_pred_opp,
        how="left",
        left_on=["id", "tm", "opp"],
        right_on=["id", "opp", "tm"],
    )

    nba_games_inseasonn_w_pred = rename_opponent_columns(nba_games_inseasonn_w_pred)
    nba_games_inseasonn_w_pred["pred_results_1_line_game"] = np.where(
        nba_games_inseasonn_w_pred["prediction_proba_df_win"]
        > nba_games_inseasonn_w_pred["prediction_proba_df_win_opp"],
        1,
        0,
    )

    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred.drop_duplicates(
        subset=["id"], keep="first"
    )
    return nba_games_inseasonn_w_pred


def save_results(
    nba_games_inseasonn_w_pred: pd.DataFrame, inference_dataset_output: str
) -> None:
    """Save the final prediction DataFrame to a CSV file."""
    final_columns = [
        "id",
        "id_season",
        "game_date",
        "tm",
        "opp",
        "results",
        "prediction_value",
        "pred_results_1_line_game",
        "prediction_proba_df_win",
        "prediction_proba_df_loose",
        "prediction_proba_df_win_opp",
        "prediction_proba_df_loose_opp",
    ]
    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred[final_columns]
    nba_games_inseasonn_w_pred.to_csv(inference_dataset_output, index=False)


def inference(
    inseason_dataset_file_path: str,
    model_columns: str,
    model_file_path: str,
    target_variable: str,
    group_cv_variable: str,
    inference_dataset_output: str,
) -> None:
    """Run inference pipeline."""

    nba_games_inseason_dataset_final, column_to_select = load_data(
        nba_games_inseason_dataset_path=inseason_dataset_file_path,
        model_columns_path=model_columns,
    )
    model = joblib.load(model_file_path)

    x_inseason, _ = prepare_data(
        nba_games_inseason_dataset_final,
        column_to_select,
        target_variable,
        group_cv_variable,
    )

    nba_games_inseasonn_w_pred = predict_and_create_dataframe(
        model,
        x_inseason,
        nba_games_inseason_dataset_final,
        target_variable,
        group_cv_variable,
    )
    nba_games_inseasonn_w_pred = add_opponent_features(nba_games_inseasonn_w_pred)

    save_results(nba_games_inseasonn_w_pred, inference_dataset_output)
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
        "--model-columns",
        dest="model_columns",
        type=str,
        default=get_models_params["model_columns"],
    )

    parser.add_argument(
        "--model-file-path",
        dest="model_file_path",
        type=str,
        default=get_models_params["model"],
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

    parser.add_argument(
        "--inference-dataset-output",
        dest="inference_dataset_output",
        type=str,
        default=inference_params["dataset"],
    )

    args = parser.parse_args()

    return args


def main():
    """Run the Pre Train Multiple Models Pipeline."""
    args = get_args()

    inference(
        inseason_dataset_file_path=args.inseason_dataset_file_path,
        model_columns=args.model_columns,
        model_file_path=args.model_file_path,
        target_variable=args.target_variable,
        group_cv_variable=args.group_cv_variable,
        inference_dataset_output=args.inference_dataset_output,
    )


if __name__ == "__main__":
    main()

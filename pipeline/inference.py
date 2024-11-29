"""Pre Train Multiple models."""

import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pipeline.utils.logs import get_logger


def rename_opponent_columns(training_df: pd.DataFrame) -> pd.DataFrame:
    """Rename opponent columns."""
    training_df.columns = training_df.columns.str.replace("_y", "_opp")
    training_df.columns = training_df.columns.str.replace("_x", "")
    return training_df


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, encoding="utf-8") as conf_file:
        return yaml.safe_load(conf_file)


def load_data(config: dict) -> tuple:
    """Load datasets and columns from configuration."""
    nba_games_inseason_dataset_final = pd.read_csv(
        config["get_inseason_dataset"]["dataset"]
    )
    columns_selected = pd.read_csv(config["get_models"]["model_columns"])
    columns_selected.columns = ["index", "columns_name"]
    column_to_select = columns_selected["columns_name"].values
    return nba_games_inseason_dataset_final, column_to_select


def prepare_data(nba_games_inseason_dataset_final: pd.DataFrame, column_to_select: list, config: dict) -> tuple:
    """Prepare input features and target variables."""
    target_column = config["inference"]["target_variable"]
    group_cv_variable = config["inference"]["group_cv_variable"]

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


def predict_and_create_dataframe(model, x_inseason: np.ndarray, nba_games_inseason_dataset_final: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Perform predictions and create the prediction DataFrame."""
    target_column = config["inference"]["target_variable"]
    group_cv_variable = config["inference"]["group_cv_variable"]

    prediction_value = model.predict(x_inseason)
    prediction_proba = model.predict_proba(x_inseason)
    prediction_proba_df = pd.DataFrame(prediction_proba, columns=["prediction_proba_df_0", "prediction_proba_df_1"])

    nba_games_inseasonn_w_pred = nba_games_inseason_dataset_final[
        [target_column, group_cv_variable, "id_season", "tm", "opp"]
    ].copy()

    nba_games_inseasonn_w_pred["prediction_proba_df_0"] = prediction_proba_df["prediction_proba_df_0"]
    nba_games_inseasonn_w_pred["prediction_proba_df_1"] = prediction_proba_df["prediction_proba_df_1"]
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
    opponent_features = ["id", "tm", "opp", "prediction_proba_df_loose", "prediction_proba_df_win"]
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

    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred.drop_duplicates(subset=["id"], keep="first")
    return nba_games_inseasonn_w_pred


def save_results(nba_games_inseasonn_w_pred: pd.DataFrame, config: dict) -> None:
    """Save the final prediction DataFrame to a CSV file."""
    final_columns = [
        "id",
        "id_season",
        "tm",
        "opp",
        "results",
        "prediction_value",
        "pred_results_1_line_game",
        "prediction_proba_df_loose",
        "prediction_proba_df_win",
        "prediction_proba_df_loose_opp",
        "prediction_proba_df_win_opp",
    ]
    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred[final_columns]
    nba_games_inseasonn_w_pred.to_csv(config["inference"]["dataset"], index=False)


def inference(config_path: str) -> None:
    """Run inference pipeline."""
    config = load_config(config_path)
    logger = get_logger("INFERENCE_STEP", log_level=config["base"]["log_level"])

    nba_games_inseason_dataset_final, column_to_select = load_data(config)
    model = joblib.load(config["get_models"]["model"])

    x_inseason, _ = prepare_data(nba_games_inseason_dataset_final, column_to_select, config)

    nba_games_inseasonn_w_pred = predict_and_create_dataframe(model, x_inseason, nba_games_inseason_dataset_final, config)
    nba_games_inseasonn_w_pred = add_opponent_features(nba_games_inseasonn_w_pred)

    save_results(nba_games_inseasonn_w_pred, config)
    logger.info("Inference Step Done")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    inference(config_path=args.config)

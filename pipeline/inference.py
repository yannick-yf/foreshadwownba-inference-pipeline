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
    """
    Rename opponent columns
    """
    training_df.columns = training_df.columns.str.replace("_y", "_opp")
    training_df.columns = training_df.columns.str.replace("_x", "")

    return training_df

def inference(config_path: dict) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    # -----------------------------------------------
    # Read daa for feature creation

    logger = get_logger("INFERENCE_STEP", log_level=config["base"]["log_level"])

    # -----------------------------------------------
    # Read input params

    target_column = config['inference']['target_variable']
    group_cv_variable = config['inference']['group_cv_variable']
    model = joblib.load(config['get_models']['model'])

    nba_games_inseason_dataset_final = pd.read_csv(
        config['get_inseason_dataset']['dataset']
        )
    columns_selected = pd.read_csv(
        config['get_models']['model_columns']
        )
    columns_selected.columns=['index', 'columns_name']

    column_to_select = columns_selected['columns_name'].values

    # ----------------------------------------------------
    # Get the proba to get the Metric evaluation per game
    # We will comapre proba to win from team 1 vs team 2
    # It leads to have 1 lign per game

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

    prediction_value = model.predict(x_inseason)

    # ----------------------------------------------------
    # Get the proba to get the Metric evaluation per game
    # We will comapre proba to win from team 1 vs team 2
    # It leads to have 1 lign per game
    prediction_proba = model.predict_proba(x_inseason)
    prediction_proba_df = pd.DataFrame(prediction_proba)
    prediction_proba_df.columns = ["prediction_proba_df_0", "prediction_proba_df_1"]

    # ---------------------------------------------

    nba_games_inseasonn_w_pred = nba_games_inseason_dataset_final[
        [
            target_column,
            group_cv_variable,
            "id_season",
            "tm",
            "opp",
        ]
    ]

    nba_games_inseasonn_w_pred["prediction_proba_df_0"] = prediction_proba_df[
        "prediction_proba_df_0"
    ].copy()
    nba_games_inseasonn_w_pred["prediction_proba_df_1"] = prediction_proba_df[
        "prediction_proba_df_1"
    ].copy()
    nba_games_inseasonn_w_pred["prediction_value"] = prediction_value

    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred.rename(
        columns={
            "prediction_proba_df_0": "prediction_proba_df_loose",
            "prediction_proba_df_1": "prediction_proba_df_win",
        }
    )

    #------------------------------------
    # Get opponent features
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
        subset=["id"], 
        keep="first"
        )
    
    #--------------------------------
    # Final column selection
    nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred[[
        'id', 'id_season', 'tm', 'opp', 
        'results', 'prediction_value', 'pred_results_1_line_game',
        'prediction_proba_df_loose',	'prediction_proba_df_win', 
        'prediction_proba_df_loose_opp',	'prediction_proba_df_win_opp'
        ]]
    
    #--------------------------------
    # Save Inseason Prediction file to CSV

    nba_games_inseasonn_w_pred.to_csv(
        config['inference']['dataset'], 
        index=False
    )

    logger.info("Inference Step Done")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    inference(config_path=args.config)

"""Pre Train Multiple models."""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, accuracy_score

from src.utils.logs import get_logger


def rename_opponent_columns(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename opponent columns
    """
    training_df.columns = training_df.columns.str.replace("_y", "_opp")
    training_df.columns = training_df.columns.str.replace("_x", "")

    return training_df


def write_bar_plot_df_from_json(report: dict, filename: str) -> pd.DataFrame:
    """
    Generate plot from json
    """
    bar_plot_data = pd.json_normalize(report)
    bar_plot_data = pd.DataFrame(
        bar_plot_data[
            [
                "cv_accuracy",
                "evaluation_2lg_accuracy",
                "evaluation_1lg_accuracy",
                "eval_vs_baseline_diff_accuracy",
            ]
        ].T
    ).reset_index()

    bar_plot_data.columns = ["name", "metric_value"]
    bar_plot_data.to_csv(filename, index=False)

    return "plot generated"


def inference(config_path: dict) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    # -----------------------------------------------
    # Read daa for feature creation

    logger = get_logger("EVALUATION_STEP", log_level=config["base"]["log_level"])

    # -----------------------------------------------
    # Read input params

    target_column = config["dummy_classifier"]["target_variable"]
    group_cv_variable = config["data_split"]["group_cv_variable"]
    metrics_path = config["evaluate"]["metrics_path"]

    logger.info("Load trained model")
    model = joblib.load("./models/model.joblib")

    test_df = pd.read_csv("./data/processed/test_dataset_fs.csv")

    list_columns_to_delete = [
        target_column,
        group_cv_variable,
        "id_season",
        "tm",
        "opp",
    ]

    logger.info("Evaluate (build report)")
    y_test = test_df.loc[:, target_column].values
    x_test = test_df.drop(list_columns_to_delete, axis=1).values

    prediction_value = model.predict(x_test)

    # ----------------------------------------------------
    # Get the proba to get the Metric evaluation per game
    # We will comapre proba to win from team 1 vs team 2
    # It leads to have 1 lign per game
    prediction_proba = model.predict_proba(x_test)
    prediction_proba_df = pd.DataFrame(prediction_proba)
    prediction_proba_df.columns = ["prediction_proba_df_0", "prediction_proba_df_1"]

    test_df_w_pred = test_df[
        [
            target_column,
            group_cv_variable,
            "id_season",
            "tm",
            "opp",
        ]
    ]

    test_df_w_pred["prediction_proba_df_0"] = prediction_proba_df[
        "prediction_proba_df_0"
    ].copy()
    test_df_w_pred["prediction_proba_df_1"] = prediction_proba_df[
        "prediction_proba_df_1"
    ].copy()
    test_df_w_pred["prediction_value"] = prediction_value

    test_df_w_pred.to_csv("./models/test_df_w_pred.csv", index=False)

    test_df_w_pred = test_df_w_pred.rename(
        columns={
            "prediction_proba_df_0": "prediction_proba_df_loose",
            "prediction_proba_df_1": "prediction_proba_df_win",
        }
    )

    opponent_features = [
        "id",
        "tm",
        "opp",
        "prediction_proba_df_loose",
        "prediction_proba_df_win",
    ]

    test_df_w_pred_opp = test_df_w_pred[opponent_features]

    test_df_w_pred = pd.merge(
        test_df_w_pred,
        test_df_w_pred_opp,
        how="left",
        left_on=["id", "tm", "opp"],
        right_on=["id", "opp", "tm"],
    )

    test_df_w_pred = rename_opponent_columns(test_df_w_pred)

    test_df_w_pred["pred_results_1_line_game"] = np.where(
        test_df_w_pred["prediction_proba_df_win"]
        > test_df_w_pred["prediction_proba_df_win_opp"],
        1,
        0,
    )

    test_df_w_pred = test_df_w_pred.drop_duplicates(subset=["id"], keep="first")

    evaluation_one_line_per_game_accuracy = round(
        accuracy_score(
            test_df_w_pred["results"], test_df_w_pred["pred_results_1_line_game"]
        ),
        3,
    )

    logger.info(
        "EVALUATION 2 LINE PER GAME ACCURACY: %s",
        round(evaluation_one_line_per_game_accuracy, 3),
    )

    # ---------------------------------------------

    evaluation_accuracy = accuracy_score(y_test, prediction_value)

    logger.info(
        "EVALUATION 2LINES PER GAME ACCURACY: %s", round(evaluation_accuracy, 3)
    )

    logger.info("Load Baseline dataset")
    # Open and read the JSON file
    with open(
        "data/reports/baseline_classifier_metrics.json", "r", encoding="utf-8"
    ) as file:
        baseline_classifier_metrics = json.load(file)

    eval_vs_baseline_accuracy = round(
        evaluation_accuracy - baseline_classifier_metrics["baseline_accuracy"], 3
    )

    logger.info(
        "EVALUATION VS BASELINE ACCURACY DIFF: %s", round(eval_vs_baseline_accuracy, 3)
    )

    report = {
        "cv_accuracy": round(cv_accuracy, 3),
        "evaluation_2lg_accuracy": round(evaluation_accuracy, 3),
        "evaluation_1lg_accuracy": round(evaluation_one_line_per_game_accuracy, 3),
        "eval_vs_baseline_diff_accuracy": eval_vs_baseline_accuracy,
        "actual": y_test,
        "predicted": prediction_value,
    }

    logger.info("Save metrics")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            obj={
                "accuracy_cv_score": report["cv_accuracy"],
                "accuracy_2lg_evaluation_score": report["evaluation_2lg_accuracy"],
                "accuracy_1lg_evaluation_score": report["evaluation_1lg_accuracy"],
                "eval_vs_baseline_diff_accuracy": report[
                    "eval_vs_baseline_diff_accuracy"
                ],
            },
            fp=file,
        )

    logger.info(
        "Accuracy & Precision metrics file saved to : {'data/reports/metrics.json'}"
    )

    # Accurcay Barplot - Report Data Processing for the plot:
    bar_plot_data_path = "./data/reports/bar_plot_data.csv"
    write_bar_plot_df_from_json(report, bar_plot_data_path)

    # ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        test_df_w_pred[target_column],
        test_df_w_pred["prediction_value"],
        normalize="true",
        cmap=plt.cm.Blues,
    )
    plt.savefig("./data/reports/ConfusionMatrix.png")

    # Save shape value plot
    explainer = shap.Explainer(model)
    shap_values = explainer(train_df.drop(list_columns_to_delete, axis=1))

    plt.figure(figsize=(10, 10))
    shap_beeswarm_path = "./data/reports/shap_beeswarm.png"
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(shap_beeswarm_path, bbox_inches="tight", dpi=100)

    plt.figure(figsize=(10, 10))
    shap.plots.bar(shap_values, show=False)
    plt.savefig("./data/reports/shap_plot_bar.png", bbox_inches="tight", dpi=100)

    logger.info(
        "Shap plots saved to : %s, './data/reports/shap_plot_bar.png'",
        shap_beeswarm_path,
    )

    logger.info("Evaluate Step Done")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", dest="config", required=True)

    args = arg_parser.parse_args()

    evaluate(config_path=args.config)

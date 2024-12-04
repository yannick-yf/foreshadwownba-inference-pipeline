import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.inference import (
    rename_opponent_columns,
    load_data,
    prepare_data,
    predict_and_create_dataframe,
    add_opponent_features,
)

class TestInference(unittest.TestCase):
    """Unit tests for inference functions."""

    def setUp(self):
        """Set up mock configuration and data."""
        self.mock_config = {
            "get_inseason_dataset": {"dataset": "mock_dataset.csv"},
            "get_models": {
                "model_columns": "mock_columns.csv",
                "model": "mock_model.pkl",
            },
            "inference": {
                "target_variable": "results",
                "group_cv_variable": "group",
                "dataset": "test_output.csv",
            },
        }

        self.mock_data = pd.DataFrame(
            {
                "id": [1, 2],
                "id_season": ["2023", "2023"],
                "tm": ["Team1", "Team2"],
                "opp": ["Team2", "Team1"],
                "results": [1, 0],
                "group": [10, 20],
                "feature1": [0.5, 0.6],
                "feature2": [0.7, 0.8],
                "game_date": ["2023-10-01", "2023-10-02"],
            }
        )

        self.selected_columns = ["feature1", "feature2"]

    def test_rename_opponent_columns(self):
        """Test renaming opponent columns."""
        df = pd.DataFrame({"col_y": [1], "col_x": [2]})
        renamed_df = rename_opponent_columns(df)
        self.assertIn("col_opp", renamed_df.columns)
        self.assertNotIn("col_y", renamed_df.columns)

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test loading datasets and selected columns."""
        mock_read_csv.side_effect = [
            self.mock_data,
            pd.DataFrame({"index": [0, 1], "columns_name": self.selected_columns}),
        ]

        dataset, columns = load_data(
            nba_games_inseason_dataset_path="mock_dataset.csv",
            model_columns_path="mock_columns.csv",
        )

        pd.testing.assert_frame_equal(dataset, self.mock_data)
        self.assertListEqual(list(columns), self.selected_columns)

    def test_prepare_data(self):
        """Test data preparation."""
        x, y = prepare_data(
            self.mock_data,
            self.selected_columns,
            target_column="results",
            group_cv_variable="group",
        )

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(x.shape[1], len(self.selected_columns))
        self.assertEqual(y.shape[0], self.mock_data.shape[0])

    @patch("pandas.DataFrame")
    def test_predict_and_create_dataframe(self, mock_dataframe):
        """Test predictions and DataFrame creation."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]
        mock_model.predict_proba.return_value = [[0.2, 0.8], [0.7, 0.3]]

        x = self.mock_data[self.selected_columns].values
        result_df = predict_and_create_dataframe(
            mock_model,
            x,
            self.mock_data,
            target_column="results",
            group_cv_variable="group",
        )

        self.assertIn("prediction_proba_df_loose", result_df.columns)
        self.assertIn("prediction_proba_df_win", result_df.columns)
        self.assertIn("prediction_value", result_df.columns)

    def test_add_opponent_features(self):
        """Test adding opponent features."""
        self.mock_data["prediction_proba_df_loose"] = [0.2, 0.7]
        self.mock_data["prediction_proba_df_win"] = [0.8, 0.3]

        updated_df = add_opponent_features(self.mock_data)

        self.assertIn("prediction_proba_df_win_opp", updated_df.columns)
        self.assertIn("prediction_proba_df_loose_opp", updated_df.columns)
        self.assertIn("pred_results_1_line_game", updated_df.columns)

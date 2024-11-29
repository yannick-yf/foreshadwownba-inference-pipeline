import pytest
import pandas as pd
import numpy as np
import yaml
import argparse
from unittest import TestCase

from unittest.mock import MagicMock
from pipeline.inference import (
    rename_opponent_columns,
    load_config,
    load_data,
    prepare_data,
    predict_and_create_dataframe,
    add_opponent_features,
    save_results,
)

class TestPreviousGamesAverageFeatures(TestCase):
    def mock_data_setup(self) -> None:
        self.nba_games_inseason = (
            "./tests/data/nba_games_inseason_dataset_final.csv"
        )
        self.selected_columns = (
            "./tests/data/columns_selected.csv"
        )

    @pytest.fixture
    def mock_config():
        """Mock configuration dictionary."""
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--config", dest="config", required=True)
        args = arg_parser.parse_args()
        config_path=args.config

        config = load_config(config_path)

        return config

    @pytest.fixture
    def mock_data(self):
        """Mock dataset and selected columns."""
        
        nba_games_inseason_dataset = pd.read_csv(self.nba_games_inseason)

        nba_games_inseason_dataset = nba_games_inseason_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        nba_games_inseason_dataset = nba_games_inseason_dataset.pipe(
            nba_games_inseason_dataset, columns_to_process=["pts_tm", "pts_opp"]
        )

        return nba_games_inseason_dataset

    def test_rename_opponent_columns():
        """Test renaming opponent columns."""
        df = pd.DataFrame({"col_y": [1], "col_x": [2]})
        renamed_df = rename_opponent_columns(df)
        assert "col_opp" in renamed_df.columns
        assert "" in renamed_df.columns

    def test_load_data(self, mock_config, mock_data, monkeypatch):
        """Test loading datasets and selected columns."""
        dataset, selected_columns = mock_data
        selected_columns = self.selected_columns

        mock_read_csv = MagicMock(
            side_effect=[
                dataset, 
                pd.DataFrame({"index": [0, 1], "columns_name": selected_columns})
                ]
            )

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        loaded_dataset, loaded_columns = load_data(mock_config)
        assert loaded_dataset.equals(dataset)
        assert list(loaded_columns) == selected_columns


    def test_prepare_data(self, mock_config, mock_data):
        """Test data preparation."""
        dataset = mock_data
        selected_columns = self.selected_columns
        x, y = prepare_data(dataset, selected_columns, mock_config)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape[1] == len(selected_columns)
        assert y.shape[0] == dataset.shape[0]


    def test_predict_and_create_dataframe(self, mock_config, mock_data, monkeypatch):
        """Test predictions and DataFrame creation."""
        dataset = mock_data
        selected_columns = self.selected_columns
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]
        mock_model.predict_proba.return_value = [[0.2, 0.8], [0.7, 0.3]]

        x = dataset[selected_columns].values
        result_df = predict_and_create_dataframe(mock_model, x, dataset, mock_config)

        assert "prediction_proba_df_loose" in result_df.columns
        assert "prediction_proba_df_win" in result_df.columns
        assert "prediction_value" in result_df.columns


    def test_add_opponent_features(self, mock_data):
        """Test adding opponent features."""
        dataset = mock_data
        dataset["prediction_proba_df_loose"] = [0.2, 0.7]
        dataset["prediction_proba_df_win"] = [0.8, 0.3]

        updated_df = add_opponent_features(dataset)

        assert "prediction_proba_df_win_opp" in updated_df.columns
        assert "prediction_proba_df_loose_opp" in updated_df.columns


    def test_save_results(self, mock_config):
        """Test saving results to a file."""
        test_df = pd.DataFrame({
            "id": [1],
            "id_season": ["2023"],
            "tm": ["Team1"],
            "opp": ["Team2"],
            "results": [1],
            "prediction_value": [1],
            "pred_results_1_line_game": [1],
            "prediction_proba_df_loose": [0.2],
            "prediction_proba_df_win": [0.8],
            "prediction_proba_df_loose_opp": [0.7],
            "prediction_proba_df_win_opp": [0.3],
        })
        output_path = "./tests/data/output.csv"

        mock_config["inference"]["dataset"] = str(output_path)
        save_results(test_df, mock_config)

        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert test_df.equals(saved_df)

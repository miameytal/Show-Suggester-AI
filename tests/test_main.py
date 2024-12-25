import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
import os
from main import (
    generate_show_details,
    retrieve_order_id,
    generate_ad,
    match_user_shows,
    get_recommendations
)

class TestTVShowRecommender(unittest.TestCase):
    @patch('openai.chat.completions.create')
    def test_generate_show_details(self, mock_openai):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"name": "Test Show", "description": "A test description"}'
        mock_openai.return_value = mock_response

        name, description = generate_show_details("Test prompt")
        
        self.assertEqual(name, "Test Show")
        self.assertEqual(description, "A test description")

    @patch.dict(os.environ, {"USE_LIGHTX_STUBS": "False"})
    @patch('requests.post')
    def test_retrieve_order_id_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"body": {"orderId": "test123"}}
        mock_post.return_value = mock_response

        order_id = retrieve_order_id("Test prompt")
        
        self.assertEqual(order_id, "test123")

    @patch.dict(os.environ, {"USE_LIGHTX_STUBS": "False"})
    @patch('requests.post')
    def test_retrieve_order_id_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        order_id = retrieve_order_id("Test prompt")
        
        self.assertIsNone(order_id)

    @patch.dict(os.environ, {"USE_LIGHTX_STUBS": "False"})
    @patch('requests.post')
    def test_generate_ad_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "body": {
                "output": "https://example.com/image.jpg",
                "status": "completed"
            }
        }
        mock_post.return_value = mock_response

        result = generate_ad("test123", max_retries=1)
        
        self.assertEqual(result, "https://example.com/image.jpg")

    @patch('pandas.read_csv')
    def test_match_user_shows(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "Title": ["Breaking Bad", "The Wire", "The Sopranos"]
        })
        mock_read_csv.return_value = mock_df

        result = match_user_shows(["Breaking Bad", "The Wire"])
        
        self.assertEqual(result, ["Breaking Bad", "The Wire"])

    @patch.dict(os.environ, {"USE_LIGHTX_STUBS": "False"})
    @patch('pickle.load')
    def test_get_recommendations(self, mock_pickle):
        # Mock embeddings data
        mock_embeddings = {
            "Show1": [0.1, 0.2, 0.3],
            "Show2": [0.2, 0.3, 0.4],
            "Show3": [0.3, 0.4, 0.5]
        }
        mock_pickle.return_value = mock_embeddings

        recommendations = get_recommendations(["Show1"])
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(all(isinstance(r, tuple) for r in recommendations))
        self.assertTrue(all(isinstance(r[0], str) and isinstance(r[1], float) 
                          for r in recommendations))

    def test_get_recommendations_empty_input(self):
        with self.assertRaises(Exception):
            get_recommendations([])

from unittest.mock import patch, mock_open
import builtins
import webbrowser

class TestAdditionalCoverage(unittest.TestCase):

    @patch("main.client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pandas.read_csv")
    def test_compute_embeddings(self, mock_read_csv, mock_file, mock_client):
        """Covers compute_embeddings by mocking CSV data and OpenAI client."""
        from main import compute_embeddings
        # Setup fake data
        mock_read_csv.return_value = pd.DataFrame({"Title": ["T1"], "Description": ["D1"]})
        
        # Instead of using a dict with a "embedding" key, create a MagicMock that has an embedding attribute:
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        fake_embedding_result = MagicMock()
        fake_embedding_result.data = [mock_embedding]  # So code can query response.data[0].embedding

        mock_client.embeddings.create.return_value = fake_embedding_result

        compute_embeddings()  # Should pickle embeddings

        mock_file.assert_called_once()  # Ensures a file was opened for writing
        mock_client.embeddings.create.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data=b"\x80\x04\x95")
    def test_load_embeddings(self, mock_file):
        """Covers load_embeddings by mocking pickle.load and verifying prints."""
        from main import load_embeddings
        with patch("pickle.load", return_value={"TitleA": [0.1, 0.2, 0.3]}), \
             patch("builtins.print") as mock_print:
            load_embeddings()
            mock_print.assert_any_call("Total number of embeddings: 1")

    @patch("main.retrieve_order_id_stub")
    @patch("main.generate_ad_stub")
    @patch("main.input", side_effect=["BadShow", "Team A", "Team B", "", "Team A, Team B", "y"])
    @patch("main.webbrowser.open")
    @patch("main.match_user_shows", return_value=["Team A", "Team B"])
    @patch("main.get_recommendations", return_value=[("Rec A", 0.9), ("Rec B", 0.8)])
    def test_main_flow_stubs(
        self, mock_get_recs, mock_match_shows, mock_browser_open, mock_input,
        mock_ad_stub, mock_order_stub
    ):
        """Covers main() using stubs for the LightX calls."""
        from main import main, should_use_lightx_stubs

        # Temporarily set stubs = True
        with patch.dict(os.environ, {"USE_LIGHTX_STUBS": "True"}):
            stub_usage = should_use_lightx_stubs()
            self.assertTrue(stub_usage)

            main()  # This runs the interactive loop with stubs

            # Our side effects ensure we eventually accept "Team A, Team B"
            # get_recommendations is mocked, so the final prints are covered
            mock_ad_stub.assert_called()  # Means generate_ad was called via stub
            mock_order_stub.assert_called()  # Means retrieve_order_id was called via stub
            mock_browser_open.assert_called()  # Means webbrowser.open lines are covered

    @patch("main.input", side_effect=["Show1, Show2", "y"])
    @patch("main.webbrowser.open")
    @patch("main.match_user_shows", return_value=["Show1", "Show2"])
    @patch("main.get_recommendations", return_value=[("Rec1", 0.7), ("Rec2", 0.6)])
    @patch("main.retrieve_order_id", return_value="test_order")
    @patch("main.generate_ad", return_value="https://example.com/ad.jpg")
    def test_main_flow_real(
        self, mock_gen_ad, mock_ret_id, mock_get_recs, mock_match_shows,
        mock_browser, mock_input
    ):
        """Covers main() with 'real' (non-stub) calls mocked, letting coverage see the branch."""
        from main import main, should_use_lightx_stubs
        with patch.dict(os.environ, {"USE_LIGHTX_STUBS": "False"}):
            stub_usage = should_use_lightx_stubs()
            self.assertFalse(stub_usage)

            main()  # Confirms we accept the matched shows and do real calls (which are mocked)

            mock_ret_id.assert_called()
            mock_gen_ad.assert_called()
            mock_browser.assert_called()

if __name__ == '__main__':
    unittest.main()
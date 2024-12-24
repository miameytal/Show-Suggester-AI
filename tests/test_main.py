import pytest
from main import match_user_shows

def test_match_user_shows():
    user_input = ["gem of throns", "lupan"]
    expected_output = ["Game of Thrones", "Lupin"]
    assert match_user_shows(user_input) == expected_output
    
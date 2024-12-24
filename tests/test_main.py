import pytest
from main import match_user_shows
from main import get_recommendations

@pytest.fixture
def user_shows():
    return ["Game of Thrones", "Lupin"]

def test_get_recommendations():
    user_shows = ["Game of Thrones", "Lupin"]
    recommendations = get_recommendations(user_shows)
    assert len(recommendations) == 5

def test_match_user_shows():
    user_input = ["gem of throns", "lupan"]
    expected_output = ["Game of Thrones", "Lupin"]
    assert match_user_shows(user_input) == expected_output
    
from thefuzz import process
import pandas as pd

def match_user_shows(user_input):
    shows = pd.read_csv("tv_shows.csv")["Title"].tolist()
    return [process.extractOne(show, shows)[0] for show in user_input]
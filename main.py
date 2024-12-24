from thefuzz import process
import pandas as pd
import openai
import pickle
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def main():
    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma:\n")
        user_shows = [show.strip() for show in user_input.split(",")]
        matched_shows = match_user_shows(user_shows)

        confirm = input(f"Making sure, do you mean {', '.join(matched_shows)}? (y/n)\n")
        if confirm.lower() == "y":
            break
        print("Sorry about that. Let's try again.")

    print("Great! Generating recommendations now...")
    recommendations = get_recommendations(matched_shows)

    print("Here are the TV shows that I think you would love:")
    for title, score in recommendations:
        print(f"{title} ({int(score * 100)}%)")

def compute_embeddings():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    shows = pd.read_csv("tv_shows.csv")
    embeddings_dict = {}

    for _, row in shows.iterrows():
        response = client.embeddings.create(
            input=row["Description"],
            model="text-embedding-ada-002"
        )
        embeddings_dict[row["Title"]] = response.data[0].embedding

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)

def match_user_shows(user_input):
    shows = pd.read_csv("tv_shows.csv")["Title"].tolist()
    return [process.extractOne(show, shows)[0] for show in user_input]

def load_embeddings():
    import pickle

    # Load the embeddings.pkl file
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Check the number of entries in the dictionary
    print(f"Total number of embeddings: {len(embeddings)}")

    # Print a sample entry
    sample_title = next(iter(embeddings))  # Get one show title
    print(f"Sample show: {sample_title}")
    print(f"Sample embedding vector (length {len(embeddings[sample_title])}): {embeddings[sample_title]}")

def get_recommendations(user_shows):
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    user_vectors = [embeddings[show] for show in user_shows]
    average_vector = np.mean(user_vectors, axis=0)

    recommendations = []
    for title, vector in embeddings.items():
        if title not in user_shows:
            similarity = cosine_similarity([average_vector], [vector])[0][0]
            recommendations.append((title, similarity))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:5]

if __name__ == "__main__":
    #compute_embeddings()
    #load_embeddings()
    main()
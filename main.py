from thefuzz import process
import pandas as pd
import openai
import pickle
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import webbrowser
import time
from usearch.index import Index


client = OpenAI()


def generate_show_details(prompt_text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Return complete JSON without truncation."},
            {"role": "user", "content": f"{prompt_text} Return perfectly valid JSON with 'name' and 'description'."}
        ],
        temperature=0.7
    )
    raw_content = response.choices[0].message.content.strip()
    # Strip possible triple backticks or language fences
    if (raw_content.startswith("```")):
        raw_content = raw_content.strip("```").strip()
        # In case it starts with "json", remove that label too.
        if (raw_content.startswith("json")):
            raw_content = raw_content[4:].strip()
    print(f"Response content after cleaning:\n{raw_content}")  # Debugging line
    data = json.loads(raw_content)
    return data['name'], data['description']

def should_use_lightx_stubs():
    return os.getenv("USE_LIGHTX_STUBS", "False").lower() in ("true", "1", "t")

def retrieve_order_id(prompt):
    if should_use_lightx_stubs():
        return retrieve_order_id_stub(prompt)
    # Real implementation:
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': os.getenv("LIGHTX_API_KEY")
    }
    data = {"textPrompt": prompt}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["body"]["orderId"] 
    else:
        return None

def generate_ad(order_id, max_retries=10, delay_in_seconds=5):
    if should_use_lightx_stubs():
        return generate_ad_stub(order_id, max_retries, delay_in_seconds)
    # Real implementation:
    url = 'https://api.lightxeditor.com/external/api/v1/order-status'
    api_key = os.getenv("LIGHTX_API_KEY")
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }
    for _ in range(max_retries):
        response = requests.post(url, headers=headers, json={"orderId": order_id})
        if response.status_code == 200:
            data = response.json()
            output = data["body"]["output"]
            if output:
                return output
            time.sleep(delay_in_seconds)
        else:
            return None
    return None

def retrieve_order_id_stub(prompt):
    print(f"Stub: retrieve_order_id called with prompt: {prompt}")
    return "mock_order_id"

def generate_ad_stub(order_id, max_retries=10, delay_in_seconds=5):
    print(f"Stub: generate_ad called with order_id: {order_id}")
    return "https://example.com/mock_ad_image.jpg"

def main():
    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show:\n")
        user_shows = [show.strip() for show in user_input.split(",")]

        if len(user_shows) < 2:
            print("Please enter more than one show.")
            continue  # This will prompt the user again

        matched_shows = match_user_shows(user_shows)

        confirm = input(f"Making sure, do you mean {', '.join(matched_shows)}? (y/n)\n")
        if confirm.lower() == "y":
            break
        print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")

    print("Great! Generating recommendations now...")
    recommendations = get_recommendations(matched_shows)

    print("Here are the TV shows that I think you would love:")
    for title, score in recommendations:
        print(f"{title} ({int(score * 100)}%)")

    # Generate creative show names and descriptions using OpenAI
    favorite_shows = ', '.join(matched_shows)
    recommended_shows = ', '.join([title for title, _ in recommendations])

    show1name, show1description = generate_show_details(
        f"In JSON format with 'name' and 'description' keys, create a name and description for a TV show inspired by the following shows: {favorite_shows}."
    )

    show2name, show2description = generate_show_details(
        f"In JSON format with 'name' and 'description' keys, create a name and description for a TV show inspired by the following shows: {recommended_shows}."
    )

    # Generate TV show ads using LightX AI Image Generator API
    ad1_url = generate_ad(retrieve_order_id(f"Create an ad for a TV show called {show1name} that is about {show1description}. The ad should be creative and engaging."))
    ad2_url = generate_ad(retrieve_order_id(f"Create an ad for a TV show called {show2name} that is about {show2description}. The ad should be creative and engaging."))

    print("\nI have also created just for you two shows which I think you would love.")
    print(f"Show #1 is based on the fact that you loved the input shows that you gave me. Its name is {show1name} and it is about {show1description}.")
    print(f"Show #2 is based on the shows that I recommended for you. Its name is {show2name} and it is about {show2description}.")
    print("Here are also the 2 TV show ads. Hope you like them!")
    print(f"Ad for {show1name}: {ad1_url}")
    print(f"Ad for {show2name}: {ad2_url}")

    webbrowser.open(ad1_url)
    webbrowser.open(ad2_url)

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
    if not user_shows:
        raise Exception("No shows provided")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    user_vectors = [embeddings[s] for s in user_shows]
    average_vector = np.mean(user_vectors, axis=0)

    index_dim = len(average_vector)
    usearch_index = Index(ndim=index_dim)

    title_to_id = {}
    id_to_title = {}
    for idx, (title, vector) in enumerate(embeddings.items()):
        title_to_id[title] = idx
        id_to_title[idx] = title
        usearch_index.add(idx, np.array(vector, dtype=np.float32))

    results = usearch_index.search(np.array(average_vector, dtype=np.float32), 10)
    recommendations_list = []
    for match in results:
        show_title = id_to_title[match.key]
        if show_title not in user_shows:
            # Cast to float to avoid np.float32 confusion in tests
            similarity = float(1.0 / (1.0 + match.distance))
            recommendations_list.append((show_title, similarity))

    recommendations_list.sort(key=lambda x: x[1], reverse=True)
    return recommendations_list[:5]

if __name__ == "__main__":
    #compute_embeddings()
    #load_embeddings()
    main()
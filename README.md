# ShowSuggesterAI

ShowSuggesterAI is a Python application that suggests TV shows based on user input and generates creative show names and descriptions using OpenAI's GPT-4 model. It also generates TV show ads using the LightX AI Image Generator API and displays them to the user.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Environmental Variables](#environmental-variables)
- [Dependencies](#dependencies)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ShowSuggesterAI.git
    cd ShowSuggesterAI
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Setup

1. **Environmental Variables:**

    Create a `.env` file in the root directory of the project and add the following environmental variables:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    LIGHTX_API_KEY=your_lightx_api_key
    USE_LIGHTX_STUBS=False  # Set to True to use stubs for LightX API during development
    ```

2. **Prepare Data:**

    Ensure you have a CSV file named `tv_shows.csv` in the root directory with the following structure:

    ```csv
    Title,Description
    Breaking Bad,A high school chemistry teacher turned methamphetamine producer...
    The Wire,A gritty drama depicting the lives of various Baltimore residents...
    ```

3. **Generate Embeddings:**

    Run the script to generate embeddings for the TV shows:
    ```sh
    python -c "from main import compute_embeddings; compute_embeddings()"
    ```

## Usage

1. Run the main application:
    ```sh
    python main.py
    ```

2. Follow the prompts to input your favorite TV shows and receive recommendations.

## Testing

1. Run the tests using `pytest`:
    ```sh
    pytest --cov=main
    ```

2. Ensure all tests pass and the coverage is satisfactory.

## Environmental Variables

- `OPENAI_API_KEY`: Your OpenAI API key.
- `LIGHTX_API_KEY`: Your LightX API key.
- `USE_LIGHTX_STUBS`: Set to `True` to use stubs for LightX API during development to avoid consuming API credits.

## Dependencies

The project requires the following dependencies, which are listed in `requirements.txt`:

- openai
- thefuzz
- scikit-learn
- numpy
- pytest
- pytest-cov
- pickle-mixin
- usearch
- requests
- colorama
- Pillow

Ensure all dependencies are installed by running:
```sh
pip install -r requirements.txt
```


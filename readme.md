# HomeMatch Application

## Introduction
HomeMatch is a real estate application designed to create personalized real estate listings. It leverages Large Language Models (LLMs) and vector databases to match real estate listings with buyer preferences and generate personalized narratives for each property to improve its attractiveness and relevance.

## Files and Directories
- `HomeMatch.py`: The main Python script.
- `listings.csv`: CSV file containing generated real estate listings.
- `example_output.txt`: Shows an example of two original listings and their improved counterparts.
- `requirements.txt`: Lists all the Python dependencies required for the project.

## Main Functions

### `generate_in_csv(llm, csv_filename)`
Generates synthetic real estate listings using a provided LLM and saves them to a CSV file. It checks if the file exists to avoid overwriting existing data.

### `load_store_in_database()`
Loads listings from a CSV file, converts them into suitable embeddings, and stores them in a vector database for semantic searching capabilities.

### `collect_buyer_preferences()`
Collects buyer preferences through natural language input and returns a structured representation of these preferences.

### `search_and_retrieve_original(db, preference_string)`
Performs a semantic search based on the user's preferences to find matching listings and retrieves the original listing from the CSV file. It returns the original listing text for the top matched listing.

### `update_listing(preferences, matching_listing)`
Generates an updated real estate listing using the data from the matching listing and the buyer's preferences, formatted according to the project's requirements.

## Getting Started
1. Install the required Python dependencies: `pip install -r requirements.txt`.
2. Run the script: `python HomeMatch.py`.

## Code expects an environment variable containing your OpenAI API key. 

export OPENAI_API_KEY='INSERT YOUR KEY HERE'

## Dependencies
- LangChain
- OpenAI GPT-3.5 Turbo (for generating and processing listings)
- Chroma (for storing and searching listings in a vector database)
- Pandas (for CSV file manipulation)
- Python 3.x

## Contribution
Contributions are welcome! Please open an issue or submit a pull request with your proposed changes.


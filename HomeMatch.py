import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import csv
import pandas as pd

# Code expects an environment variable containing your OpenAI API key
# export OPENAI_API_KEY='INSERT YOUR KEY HERE'


model_name = 'gpt-3.5-turbo'
llm = ChatOpenAI(model_name=model_name, temperature=0.7, max_tokens=700)

def generate_in_csv(llm, csv_filename="listings.csv"):
    '''
    Function to generate and store real estate listings. If file exists no new listings will be generated

    - ./listings.csv' : filename for listing file. pls delete if you want new listings

    '''

    # Check if the file already exists
    if os.path.exists(csv_filename):
        print(f"\n{csv_filename} already exists. No new listings will be generated.")
        return

    listings = []
    for _ in range(10):  # Adjust as necessary for the number of listings
        prompt = "Create a real estate listing with the following attributes: neighborhood, price, bedrooms, bathrooms, house size, and a detailed description."
        response = llm.invoke(prompt)
        listings.append(response.content)  # Assuming response.content contains the listing text
    
    # Open a CSV file for writing
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Listing"])  # Writing header row
        
        for listing in listings:
            writer.writerow([listing])  # Writing each listing in its row

    print(f"\nListings have been generated and stored in {csv_filename}.")

# Store listings in a vector database
def load_store_in_database():
    ''' 
    Function loads listings from file and stores their embeddings in a database
    '''
    
    loader = CSVLoader(file_path='./listings.csv')
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(split_docs, embeddings)

    return db

def collect_buyer_preferences():
    """
    Hardcoded preferences for natural language input.
    Modify as you see fit.
    """
    preference_string = " ".join([
    "I prefer a single-family home with a yard for outdoor activities.",
    "Being close to work and quality schools is crucial for me, preferably within a 20-minute commute.",
    "I'm looking for properties that offer at least 1,500 square feet to accommodate my family comfortably.",
    "I love modern homes with minimalist design, large windows for natural light, and energy-efficient systems.",
    "Having community amenities within walking distance is important for my lifestyle, especially parks, a gym, and a few dining options.",
    "I want to live in Brooklyn"
    ])
    
    return(preference_string)

def search_and_retrieve_original(db, preference_string):
    """
    Performs a semantic search based on user preferences to find matching listings,
    then retrieves and prints the original listings from the CSV file for all matches.

    Parameters:
    - db: The vector database instance.
    - preference_string: A string representing the user's preferences.
    - Modify 'k' to retrieve more matches 

    Returns:
    A list of dictionaries, each containing the original listing text and its row index.
    """
    print(f"\nUser preferences: {preference_string}\n")

    # Perform the search
    matched_listings = db.similarity_search(preference_string, k=2)  # Increase k for more matches

    matching_listings = []

    if matched_listings:
        df = pd.read_csv('./listings.csv')
        for match in matched_listings:
            row_index = match.metadata['row']
            matching_listing = df.iloc[row_index]['Listing']
            print(f"\nMatched Listing:\n\n{matching_listing}")
            matching_listings.append({"text": matching_listing, "row": row_index})

    if not matching_listings:
        print("No matching listings found.")

    return matching_listings


# Have LLM create new listing
def update_listing(preferences, matching_listings):
    """
    Updates each matched listing with a personalized description based on user preferences.

    Parameters:
    - preferences: The buyer's preferences.
    - matching_listings: A list of dictionaries, each containing the original listing text and its row index.

    Returns:
    A list of updated listings with personalized descriptions.
    """
    updated_listings = []
    
    for matching_listing in matching_listings:
        prompt = f"""
        Given the original_listing below and the user's preferences 
        create an updated real estate listing using the data from the original listing.
        Use the format as described below.

        ***Do not change the factual details (Neighborhood, Price, Bedrooms, Bathrooms, House Size). Only update the general description to align with the user's preferences.***

        Fill in the fields using the exact data from the original listing
        {matching_listing["text"]}
        Neighborhood:
        Price: 
        Bedrooms: 
        Bathrooms: 
        House Size: 

        Create a new description based on:
        {preferences}
     
        """

        response = llm.invoke(prompt)
        updated_listing_text = response.content.strip()
        updated_listings.append(updated_listing_text)
        
    return updated_listings

# 7. Main application logic
def main():
    # Generate listings
    generate_in_csv(llm)

    # Load and store listings in a database
    db = load_store_in_database()
    
    # Collect buyer preferences
    preferences = collect_buyer_preferences()
    
    # Search for matches
    matching_listings = search_and_retrieve_original(db, preferences)

    # Update original listing
    updated_listings = update_listing(preferences, matching_listings)
    
    # Print all newly created listings
    for updated_listing in updated_listings:
        print("\nUpdated and personalized listing:\n")
        print(updated_listing)
        print("\n---\n")  # Separator for readability

if __name__ == "__main__":
    main()

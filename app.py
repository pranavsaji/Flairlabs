import os
import pandas as pd
import re
from groq import Groq 
import pandasql as psql
import streamlit as st
import sqlparse
import base64


# ============================================
# Helper Functions
# ============================================


def extract_bed_bath_combinations(sql_query):
    """
    Extracts bed-bath combinations from the SQL query.
    Example: Extracts "beds = 2 AND baths = 2" as {'beds': 2, 'baths': 2}
    """
    bed_bath_conditions = []
    
    # Updated Regex pattern to extract bed-bath combinations from SQL query
    # This pattern is flexible and accounts for possible variations
    bed_bath_pattern = r"\bbeds?\s*=\s*(\d+)\s*AND\s*\bbaths?\s*=\s*(\d+)\b"
    print("bed_bath_pattern is", bed_bath_pattern)  # Debugging statement
    
    # Find all bed-bath combinations in the SQL query
    bed_bath_matches = re.findall(bed_bath_pattern, sql_query, re.IGNORECASE)
    print("bed_bath_matches is", bed_bath_matches)  # Debugging statement
    
    # Store each combination as a dictionary with 'beds' and 'baths'
    for match in bed_bath_matches:
        bed_bath_conditions.append({
            'beds': int(match[0]),
            'baths': int(match[1])
        })
    
    return bed_bath_conditions

def extract_first_school_rating(school_ratings):
    """
    Extracts the first school rating from the 'school_ratings' column.
    """
    if isinstance(school_ratings, str):
        # Split the string and try to take the first value
        ratings = school_ratings.split()
        if ratings:
            try:
                # Return the first value as an integer
                return int(ratings[0])
            except ValueError:
                return None
    elif isinstance(school_ratings, (int, float)):
        return school_ratings
    return None

def check_feature_in_description(desc, feature):
    """
    Checks if a specific feature exists within the 'neighborhood_desc' column.
    """
    # Use word boundaries and allow for singular/plural forms
    pattern = re.compile(rf'\b{re.escape(feature)}s?\b', re.IGNORECASE)
    return bool(pattern.search(desc)) if isinstance(desc, str) else False

def extract_additional_features(traits, key_phrases, possible_features, feature_synonyms):
    """
    Extracts additional features from traits and key phrases.
    Only includes features present in the possible_features list, mapping synonyms.
    """
    extracted_features = []
    
    # Combine traits and key phrases into a single string for searching
    combined_text = ' '.join(traits + key_phrases).lower()
    
    for feature, standard_feature in feature_synonyms.items():
        # Use regex word boundaries to ensure accurate matching
        feature_pattern = rf'\b{re.escape(feature)}\b'
        if re.search(feature_pattern, combined_text):
            if standard_feature in possible_features and standard_feature not in extracted_features:
                extracted_features.append(standard_feature)
    
    # Now, check for features without synonyms
    for feature in possible_features:
        if feature not in feature_synonyms.values():  # To avoid duplication
            feature_pattern = rf'\b{re.escape(feature)}\b'
            if re.search(feature_pattern, combined_text):
                if feature not in extracted_features:
                    extracted_features.append(feature)
    
    return extracted_features

# Define the list of possible additional features (use base terms where appropriate)
possible_features = [
    'good school nearby', 'excellent school nearby', 'pool', 'restaurant', 'gym', 'rooftop access',
    'home theater', 'wine cellar', 'large garden', 'high ceiling', 'hardwood floors',
    'finished basement', 'garage', 'exposed brick walls', 'spacious backyard',
    'solar panels', 'panoramic city views', 'private elevator', 'fireplace'
]

# Define the feature synonyms mapping (map variations to standard feature names)
feature_synonyms = {
    'restaurant nearby': 'restaurant',
    'near a restaurant': 'restaurant',
    'restaurants nearby': 'restaurant',
    'restaurants': 'restaurant',
    'near restaurants': 'restaurant',
    'highly-rated school': 'good school nearby',
    'good school district':'good school nearby',
    'high-ranked school nearby':'good school nearby',
    'excellent school district':'excellent school nearby',
    'excellent school': 'excellent school nearby',
    # Add more synonyms as needed
}
# Extract dynamic columns from the SQL query
# def extract_columns_from_sql(sql_query):
#     dynamic_columns = []
#     bed_bath_conditions = []
#     city_conditions = []
#     feature_conditions = []

#     # Regex to find bed-bath combinations
#     bed_bath_pattern = r"(\d+)\s*bed\s*(\d+)\s*bath"
#     city_pattern = r"city\s+LIKE\s+'%([^%]+)%'"
#     feature_pattern = r"neighborhood_desc\s+LIKE\s+'%([^%]+)%'"

#     # Find all bed-bath combinations
#     bed_bath_matches = re.findall(bed_bath_pattern, sql_query, re.IGNORECASE)
#     for match in bed_bath_matches:
#         bed_bath_conditions.append({
#             'beds': int(match[0]),
#             'baths': int(match[1])
#         })

#     # Find all cities mentioned in the query
#     city_matches = re.findall(city_pattern, sql_query, re.IGNORECASE)
#     for match in city_matches:
#         city_conditions.append(match)

#     # Find all features mentioned in the query
#     feature_matches = re.findall(feature_pattern, sql_query, re.IGNORECASE)
#     for match in feature_matches:
#         feature_conditions.append(match)

#     return dynamic_columns, bed_bath_conditions, city_conditions, feature_conditions

# def check_feature_in_description(desc, feature):
#     pattern = re.compile(rf'\b{feature}s?\b', re.IGNORECASE)
#     return bool(pattern.search(desc)) if isinstance(desc, str) else False



def initialize_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        return None
    return Groq(api_key=api_key)

# ================================
# Load and Preprocess Zillow Data
# ================================
@st.cache_data
def load_data(file_path='Zillow_Data.csv'):
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at path: {file_path}")
        return None
    df = pd.read_csv(file_path)
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
    else:
        st.warning("'price' column not found in the dataset.")

    # Convert 'beds' and 'baths' to numeric
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
    # Convert 'city' and 'state' to title and upper case for consistent matching
    df['city'] = df['city'].str.title()
    df['state'] = df['state'].str.upper()
    # Ensure 'zip_code' is string
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str)
    else:
        st.warning("The dataset does not contain a 'zip_code' column.")
    return df

from nltk.corpus import stopwords
import re

# List of stopwords to remove from traits (e.g., "has", "a", "the", "is near")
# List of phrases and words to remove from traits
removal_phrases = ['has', 'a', 'the', 'an', 'is near']

def extract_feature_from_trait(trait):
    """
    Extracts the relevant feature from the trait string by removing stopwords and matching 
    remaining words against the features_list.
    """
    # Tokenize the trait and remove unwanted words/phrases
    for phrase in removal_phrases:
        trait = trait.replace(phrase, '').strip()
    
    # Tokenize and clean up the trait
    tokens = [word.lower() for word in re.split(r'\W+', trait) if word]

    # Join tokens back to form the cleaned trait
    cleaned_trait = ' '.join(tokens)

    # Return the cleaned trait
    return cleaned_trait
def get_unique_cities(df):
    """
    Extract a sorted list of unique cities from the DataFrame.
    """
    return sorted(df['city'].dropna().unique())

# ====================================
# Load and Preprocess Broker Data
# ====================================
@st.cache_data
def load_broker_data(file_path='broker_data.csv'):
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at path: {file_path}")
        return None
    df = pd.read_csv(file_path)
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    # Convert 'zip_code' to string to preserve leading zeros
    df['zip_code'] = df['zip_code'].astype(str)
    # Convert 'city' and 'state' to title and upper case for consistent matching
    df['city'] = df['city'].str.title()
    df['state'] = df['state'].str.upper()
    return df

# ============================================
# Function to Get Completion from Groq API
# ============================================
def get_groq_completion(client, prompt, model="llama3-8b-8192", max_tokens=1500, temperature=0, stop=None):
    """
    Use Groq API to get a completion based on the provided prompt.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return ""

# ============================================
# Function to Extract SQL Code from Response
# ============================================
def extract_sql_code(response_text):
    """
    Extract SQL code from the assistant's response.
    Since the response contains only the SQL query, return it directly.
    """
    sql_code = response_text.strip()
    # Optionally, ensure the SQL starts with SELECT
    return sql_code if sql_code.upper().startswith("SELECT") else ""

# ============================================
# Function to Generate SQL Query 
# ============================================
def generate_sql_groq(client, query, user_intent, traits, key_phrases, cities, max_retries=5):
    """
    Generate a SQL query based on the user's natural language query using Groq.
    Incorporates user intent, traits, and key phrases into the SQL generation process.
    Attempts up to `max_retries` times if extraction fails.
    
    Parameters:
    - client: Initialized Groq client.
    - query: User's natural language query.
    - user_intent: Extracted user intent.
    - traits: Extracted traits.
    - key_phrases: Extracted key phrases.
    - cities: List of unique cities from the CSV.
    - max_retries: Number of retry attempts for generating SQL.
    
    Returns:
    - SQL query string or empty string if failed.
    """
    # Convert cities list to a readable string
    cities_str = ", ".join([f"'{city}'" for city in cities])

    prompt = (
        "You are a SQL assistant specialized in real estate data. "
        "Based on the user's natural language query, user intent, traits, and key phrases, generate an accurate SQL query to search the properties. "
        "Map any broad location terms in the user's query to specific cities from the provided list. "
        "Ensure that all property features and filters mentioned in the query are accurately represented in the SQL. "
        "Use the following CSV columns for the 'zillow_data' table: 'price', 'beds', 'baths', 'area', 'listing_agent', 'year_built', "
        "'property_tax', 'school_ratings', 'neighborhood_desc', 'broker', 'city', 'state', 'zip_code',hoa_fees "
        "Use LIKE operators for string matching and ensure numeric comparisons are correctly handled. "
        "If the user specifies a broad location like 'Bay Area,' map it to specific cities such as 'San Francisco,' 'Oakland,' etc. "
        "If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'. "
        "If no location is provided, don't add it as a filtering criterion. "
        "If certain values are not provided in the query, don't add them in the SQL query. "
        "If school rating is given as good in the query, use SQL condition as school_ratings >= 7. "
        "If school rating is given as excellent in the query, use SQL condition as school_ratings >= 8. "
        "When multiple conditions exist within the same column, combine them using OR and enclose them in parentheses. "
        "Combine conditions across different columns using AND. "
        "IMPORTANT: For all property features (e.g., 'pool', 'sea_view', 'hoa_fees', 'gym', 'rooftop_access', 'home_theater', 'wine_cellar', 'large_garden', 'high_ceiling', 'hardwood_floors', 'finished_basement', 'garage', 'exposed_brick_walls', 'spacious_backyard', 'solar_panels', 'panoramic_city_views', 'private_elevator', 'fireplace', 'swimming_pool'.etc), search within the 'neighborhood_desc' column using LIKE operators instead of using dedicated feature columns. "
        "Do NOT use separate columns like 'pool', 'gym', etc., for filtering features. Instead, encapsulate all feature-related filters within 'neighborhood_desc'. "
        "Return only the SQL query as plain text without any explanations, code fences, backticks, markdown formatting, or additional text.\n\n"

        "### Available Cities:\n"
        f"{cities_str}\n\n"

        "### Example 1:\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards.\n"
        "**Traits:**\n"
        "- is a house\n"
        "  has 3 bed, 2 bath\n"
        "  is in San Francisco\n"
        "- has a big backyard\n"
        "**Key Phrases:**\n"
        "3 bedroom house\n"
        "big backyard\n"
        "San Francisco real estate\n"
        "spacious home\n"
        "outdoor space\n"
        "family-friendly neighborhood\n"
        "gardening space\n"
        "entertainment area\n"
        "pet-friendly home\n"
        "modern amenities\n\n"
        "**User Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND city LIKE '%San Francisco%' AND (neighborhood_desc LIKE '%big backyard%');\n\n"

        
        "### Example 2:\n"
        "**User Intent:** The user is looking for houses in the Bay Area that meet specific criteria: having at least 4 bedrooms, at least 2,500 sqft, priced under $2 million, and good school ratings. They likely prioritize spacious living areas, affordability, and quality education for their family.\n"
        "**Traits:**\n"
        "- is a house\n"
        "  has at least 4 bed, at least 2 bath\n"
        "  is in the Bay Area\n"
        "- has at least 2,500 sqft\n"
        "- is priced under $2,000,000\n"
        "- has excellent school ratings\n"
        "**Key Phrases:**\n"
        "4 bedroom house\n"
        "2500 sqft\n"
        "under 2 million\n"
        "excellent school ratings\n"
        "Bay Area real estate\n"
        "spacious living\n"
        "affordable housing\n"
        "quality education\n"
        "family-friendly neighborhood\n"
        "modern amenities\n\n"
        "**User Query:** \"Looking for houses in the Bay Area with at least 4 bedrooms, at least 2,500 sqft, priced under $2 million, and excellent school ratings.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds >= 4 AND area >= 2500 AND price <= 2000000 AND school_ratings >= 7 AND (city LIKE '%San Francisco%' OR city LIKE '%Oakland%' OR city LIKE '%San Jose%' OR city LIKE '%Silicon Valley%');\n\n"

        "### Example 3:\n"
        "**User Intent:** The user wants a four-bedroom house in Miami with a pool and sea view, under a budget of $2 million.\n"
        "**Traits:**\n"
        "- is a house\n"
        "  has 4 bed, 3 bath\n"
        "  is in Miami\n"
        "- has a pool\n"
        "- has a sea view\n"
        "- is priced under $2,000,000\n"
        "**Key Phrases:**\n"
        "4 bedroom house\n"
        "pool\n"
        "sea view\n"
        "Miami real estate\n"
        "luxury home\n"
        "waterfront property\n"
        "family-friendly neighborhood\n"
        "modern design\n"
        "spacious backyard\n"
        "gated community\n\n"
        "**User Query:** \"Looking for a 4 bed 3 bath house in Miami with a pool and sea view, priced under 2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 4 AND baths = 3 AND price <= 2000000 AND city LIKE '%Miami%' AND (neighborhood_desc LIKE '%pool%' OR neighborhood_desc LIKE '%sea view%');\n\n"

        "### Example 4:\n"
        "**User Intent:** The user is searching for a three-bedroom townhouse in Denver with low HOA fees and property taxes, priced below $700,000.\n"
        "**Traits:**\n"
        "- is a townhouse\n"
        "  has 3 bed, 2 bath\n"
        "  is in Denver\n"
        "- has low HOA fees\n"
        "- has low property taxes\n"
        "- is priced under $700,000\n"
        "**Key Phrases:**\n"
        "3 bedroom townhouse\n"
        "low HOA fees\n"
        "low property taxes\n"
        "Denver real estate\n"
        "budget-friendly\n"
        "family-friendly\n"
        "spacious living\n"
        "modern amenities\n"
        "central location\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath townhouse in Denver with low HOA fees and property taxes, priced under 700,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 700000 AND city LIKE '%Denver%' AND (neighborhood_desc LIKE '%low HOA fees%' OR neighborhood_desc LIKE '%low property taxes%');\n\n"

        "### Example 5:\n"
        "**User Intent:** The user wants a two-bedroom condo in San Francisco with a gym and rooftop access, priced below $1.2 million.\n"
        "**Traits:**\n"
        "- is a condo\n"
        "  has 2 bed, 2 bath\n"
        "  is in San Francisco\n"
        "- has a gym\n"
        "- has rooftop access\n"
        "- is priced under $1,200,000\n"
        "**Key Phrases:**\n"
        "2 bedroom condo\n"
        "gym\n"
        "rooftop access\n"
        "San Francisco real estate\n"
        "modern amenities\n"
        "urban living\n"
        "secure building\n"
        "pet-friendly\n"
        "spacious interiors\n"
        "high-rise building\n\n"
        "**User Query:** \"Looking for a 2 bed 2 bath condo in San Francisco with a gym and rooftop access, priced under 1.2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 2 AND price <= 1200000 AND city LIKE '%San Francisco%' AND (neighborhood_desc LIKE '%gym%' OR neighborhood_desc LIKE '%rooftop access%');\n\n"

        "### Example 6:\n"
        "**User Intent:** The user is searching for a five-bedroom villa in Los Angeles with a home theater, wine cellar, and large garden, priced below $3 million.\n"
        "**Traits:**\n"
        "- is a villa\n"
        "  has 5 bed, 4 bath\n"
        "  is in Los Angeles\n"
        "- has a home theater\n"
        "- has a wine cellar\n"
        "- has a large garden\n"
        "- is priced under $3,000,000\n"
        "**Key Phrases:**\n"
        "5 bedroom villa\n"
        "home theater\n"
        "wine cellar\n"
        "large garden\n"
        "Los Angeles real estate\n"
        "luxury amenities\n"
        "spacious property\n"
        "gated community\n"
        "waterfront property\n"
        "exclusive neighborhood\n\n"
        "**User Query:** \"Looking for a 5 bed 4 bath villa in Los Angeles with a home theater, wine cellar, and large garden, priced under 3 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 5 AND baths = 4 AND price <= 3000000 AND city LIKE '%Los Angeles%' AND (neighborhood_desc LIKE '%home theater%' OR neighborhood_desc LIKE '%wine cellar%' OR neighborhood_desc LIKE '%large garden%');\n\n"

        "### Example 7:\n"
        "**User Intent:** The user wants a studio apartment in Redwood City with high ceilings and hardwood floors, priced below $900,000.\n"
        "**Traits:**\n"
        "- is a studio apartment\n"
        "  has 1 bed, 1 bath\n"
        "  is in Redwood City\n"
        "- has high ceilings\n"
        "- has hardwood floors\n"
        "- is priced under $900,000\n"
        "**Key Phrases:**\n"
        "studio apartment\n"
        "high ceilings\n"
        "hardwood floors\n"
        "Redwood City real estate\n"
        "modern design\n"
        "urban living\n"
        "pet-friendly\n"
        "open floor plan\n"
        "luxury finishes\n"
        "secure building\n\n"
        "**User Query:** \"Looking for a studio apartment in Redwood with high ceilings and hardwood floors, priced under 900,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 1 AND baths = 1 AND price <= 900000 AND city LIKE '%Redwood City%' AND (neighborhood_desc LIKE '%high ceilings%' OR neighborhood_desc LIKE '%hardwood floors%');\n\n"

        "### Example 8:\n"
        "**User Intent:** The user is searching for a three-bedroom duplex in Chicago with a finished basement and garage, priced below $1.5 million.\n"
        "**Traits:**\n"
        "- is a duplex\n"
        "  has 3 bed, 2 bath\n"
        "  is in Chicago\n"
        "- has a finished basement\n"
        "- has a garage\n"
        "- is priced under $1,500,000\n"
        "**Key Phrases:**\n"
        "3 bedroom duplex\n"
        "finished basement\n"
        "garage\n"
        "Chicago real estate\n"
        "spacious living\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "secure property\n"
        "close to schools\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath duplex in Chicago with a finished basement and garage, priced under 1.5 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 1500000 AND city LIKE '%Chicago%' AND (neighborhood_desc LIKE '%finished basement%' OR neighborhood_desc LIKE '%garage%');\n\n"

        "### Example 9:\n"
        "**User Intent:** The user wants a two-bedroom loft in Boston with exposed brick walls and large windows, priced below $1.1 million.\n"
        "**Traits:**\n"
        "- is a loft\n"
        "  has 2 bed, 1 bath\n"
        "  is in Boston\n"
        "- has exposed brick walls\n"
        "- has large windows\n"
        "- is priced under $1,100,000\n"
        "**Key Phrases:**\n"
        "2 bedroom loft\n"
        "exposed brick walls\n"
        "large windows\n"
        "Boston real estate\n"
        "industrial design\n"
        "open floor plan\n"
        "modern amenities\n"
        "urban living\n"
        "pet-friendly\n"
        "high ceilings\n\n"
        "**User Query:** \"Looking for a 2 bed 1 bath loft in Boston with exposed brick walls and large windows, priced under 1.1 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 1 AND price <= 1100000 AND city LIKE '%Boston%' AND (neighborhood_desc LIKE '%exposed brick walls%' OR neighborhood_desc LIKE '%large windows%');\n\n"

        "### Example 10:\n"
        "**User Intent:** The user is searching for a four-bedroom ranch-style house in Phoenix with a spacious backyard and solar panels, priced below $850,000.\n"
        "**Traits:**\n"
        "- is a ranch-style house\n"
        "  has 4 bed, 3 bath\n"
        "  is in Phoenix\n"
        "- has a spacious backyard\n"
        "- has solar panels\n"
        "- is priced under $850,000\n"
        "**Key Phrases:**\n"
        "4 bedroom ranch-style house\n"
        "spacious backyard\n"
        "solar panels\n"
        "Phoenix real estate\n"
        "energy-efficient\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "secure property\n"
        "low maintenance\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 4 bed 3 bath ranch-style house in Phoenix with a spacious backyard and solar panels, priced under 850,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 4 AND baths = 3 AND price <= 850000 AND city LIKE '%Phoenix%' AND (neighborhood_desc LIKE '%spacious backyard%' OR neighborhood_desc LIKE '%solar panels%');\n\n"

        "### Example 11:\n"
        "**User Intent:** The user wants a three-bedroom bungalow in Portland with a gourmet kitchen and hardwood floors, priced below $950,000.\n"
        "**Traits:**\n"
        "- is a bungalow\n"
        "  has 3 bed, 2 bath\n"
        "  is in Portland\n"
        "- has a gourmet kitchen\n"
        "- has hardwood floors\n"
        "- is priced under $950,000\n"
        "**Key Phrases:**\n"
        "3 bedroom bungalow\n"
        "gourmet kitchen\n"
        "hardwood floors\n"
        "Portland real estate\n"
        "modern amenities\n"
        "family-friendly neighborhood\n"
        "spacious living areas\n"
        "secure property\n"
        "pet-friendly\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath bungalow in Portland with a gourmet kitchen and hardwood floors, priced under 950,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 950000 AND city LIKE '%Portland%' AND (neighborhood_desc LIKE '%gourmet kitchen%' OR neighborhood_desc LIKE '%hardwood floors%');\n\n"

        "### Example 12:\n"
        "**User Intent:** The user is searching for a duplex in Houston with energy-efficient appliances and a home office, priced below $1.3 million.\n"
        "**Traits:**\n"
        "- is a duplex\n"
        "  has 2 bed, 2 bath per unit\n"
        "  is in Houston\n"
        "- has energy-efficient appliances\n"
        "- has a home office\n"
        "- is priced under $1,300,000\n"
        "**Key Phrases:**\n"
        "duplex\n"
        "energy-efficient appliances\n"
        "home office\n"
        "Houston real estate\n"
        "modern amenities\n"
        "spacious interiors\n"
        "family-friendly neighborhood\n"
        "secure property\n"
        "pet-friendly\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a duplex in Houston with energy-efficient appliances and a home office, priced under 1.3 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE type = 'duplex' AND price <= 1300000 AND city LIKE '%Houston%' AND (neighborhood_desc LIKE '%energy-efficient appliances%' OR neighborhood_desc LIKE '%home office%');\n\n"

        "### Example 13:\n"
        "**User Intent:** The user wants a five-bedroom farmhouse in Nashville with a barn, landscaped garden, and solar energy system, priced below $2.2 million.\n"
        "**Traits:**\n"
        "- is a farmhouse\n"
        "  has 5 bed, 4 bath\n"
        "  is in Nashville\n"
        "- has a barn\n"
        "- has a landscaped garden\n"
        "- has a solar energy system\n"
        "- is priced under $2,200,000\n"
        "**Key Phrases:**\n"
        "5 bedroom farmhouse\n"
        "barn\n"
        "landscaped garden\n"
        "solar energy system\n"
        "Nashville real estate\n"
        "luxury amenities\n"
        "spacious property\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "energy-efficient\n\n"
        "**User Query:** \"Looking for a 5 bed 4 bath farmhouse in Nashville with a barn, landscaped garden, and solar energy system, priced under 2.2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 5 AND baths = 4 AND price <= 2200000 AND city LIKE '%Nashville%' AND (neighborhood_desc LIKE '%barn%' OR neighborhood_desc LIKE '%landscaped garden%' OR neighborhood_desc LIKE '%solar energy system%');\n\n"

        "### Example 14:\n"
        "**User Intent:** The user is searching for a two-bedroom penthouse in Dallas with panoramic city views and a private elevator, priced below $1.8 million.\n"
        "**Traits:**\n"
        "- is a penthouse\n"
        "  has 2 bed, 2 bath\n"
        "  is in Dallas\n"
        "- has panoramic city views\n"
        "- has a private elevator\n"
        "- is priced under $1,800,000\n"
        "**Key Phrases:**\n"
        "2 bedroom penthouse\n"
        "panoramic city views\n"
        "private elevator\n"
        "Dallas real estate\n"
        "luxury amenities\n"
        "high-rise living\n"
        "secure building\n"
        "modern design\n"
        "spacious interiors\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 2 bed 2 bath penthouse in Dallas with panoramic city views and a private elevator, priced under 1.8 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 2 AND price <= 1800000 AND city LIKE '%Dallas%' AND (neighborhood_desc LIKE '%panoramic city views%' OR neighborhood_desc LIKE '%private elevator%');\n\n"

        "### Example 15:\n"
        "**User Intent:** The user wants a three-bedroom colonial-style house in Philadelphia with a fireplace, home theater, and swimming pool, priced below $1.4 million.\n"
        "**Traits:**\n"
        "- is a colonial-style house\n"
        "  has 3 bed, 2 bath\n"
        "  is in Philadelphia\n"
        "- has a fireplace\n"
        "- has a home theater\n"
        "- has a swimming pool\n"
        "- is priced under $1,400,000\n"
        "**Key Phrases:**\n"
        "3 bedroom colonial-style house\n"
        "fireplace\n"
        "home theater\n"
        "swimming pool\n"
        "Philadelphia real estate\n"
        "modern amenities\n"
        "spacious backyard\n"
        "family-friendly neighborhood\n"
        "secure property\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath colonial-style house in Philadelphia with a fireplace, home theater, and swimming pool, priced under 1.4 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 1400000 AND city LIKE '%Philadelphia%' AND (neighborhood_desc LIKE '%fireplace%' OR neighborhood_desc LIKE '%home theater%' OR neighborhood_desc LIKE '%swimming pool%');\n\n"

        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Traits:**\n"
        f"{traits}\n\n"
        "**Key Phrases:**\n"
        f"{key_phrases}\n\n"
        "**User Query:**\n"
        f"\"{query}\"\n\n"
        "**SQL Query:**"
    )

    for attempt in range(1, max_retries + 1):
        response = get_groq_completion(client, prompt, max_tokens=1500, temperature=0)
        sql_query = extract_sql_code(response)
        if sql_query:
            return sql_query.strip()
        else:
            st.warning(f"Attempt {attempt} to extract SQL code failed.")

    # After all retries failed
    st.error("Could not generate a valid SQL query after multiple attempts.")
    return ""


# ============================================
# Function to Generate Property Keywords
# ============================================
# def get_property_keywords_groq(client, query, user_intent, traits, key_phrases, sql_query):
#     """
#     Generate PropertyKeywords based on the query, user intent, traits, key phrases, and SQL query.
    
#     Parameters:
#     - client: Initialized Groq client.
#     - query: User's natural language query.
#     - user_intent: Extracted user intent.
#     - traits: Extracted traits.
#     - key_phrases: Extracted key phrases.
#     - sql_query: Generated SQL query.
    
#     Returns:
#     - PropertyKeywords string or empty string if extraction fails.
#     """
#     prompt = (
#         "Analyze the following real estate query, user intent, traits, key phrases, and SQL query to extract the values used for each column in the SQL statement."
#         " Format the output as a comma-separated list in the format 'Column: Value'. "
#         "Ensure that each value corresponds accurately to the SQL query."
#         "Do not right content like Here is the output in the format 'Column: Value':, only give precise output without additional text"
#         "If some location or keyword is incomplete,fill them with most appropriate value from the data, for eg: replace redwood with Redwood City"
#         " Do not include any explanations or additional text.\n\n"

#         "### Query:\n"
#         f"\"{query}\"\n\n"

#         "### User Intent:\n"
#         f"{user_intent}\n\n"

#         "### Traits:\n"
#         f"{', '.join(traits)}\n\n"

#         "### Key Phrases:\n"
#         f"{', '.join(key_phrases)}\n\n"

#         "### SQL Query:\n"
#         f"{sql_query}\n\n"

#         "### PropertyKeywords:"
#     )

#     response = get_groq_completion(client, prompt, temperature=0, max_tokens=500)
#     return response.strip()

def get_property_keywords_groq(client, query, user_intent, traits, key_phrases, sql_query):
    """
    Generate PropertyKeywords based on the query, user intent, traits, key phrases, and SQL query.
    
    Parameters:
    - client: Initialized Groq client.
    - query: User's natural language query.
    - user_intent: Extracted user intent.
    - traits: Extracted traits.
    - key_phrases: Extracted key phrases.
    - sql_query: Generated SQL query.
    
    Returns:
    - PropertyKeywords string or empty string if extraction fails.
    """
    prompt = (
        
        "Analyze the following real estate query, user intent, traits, key phrases, and SQL query to extract the values used for each column in the SQL statement."
        "Do not miss any content or value from sql."
        " Format the output as a comma-separated list in the format 'Column: Value'.But dont add any extra piece of text "
        "Ensure that each value corresponds accurately to the SQL query."
        "IMPORTANT:  Only give precise output in the format given without any additional text or tokens"
        "If some location or keyword is incomplete,fill them with most appropriate value from the data, for eg: replace redwood with Redwood City"
        " Do not include any explanations or additional text.\n\n"
    

        "### Query:\n"
        f"\"{query}\"\n\n"

        "### User Intent:\n"
        f"{user_intent}\n\n"

        "### Traits:\n"
        f"{', '.join(traits)}\n\n"

        "### Key Phrases:\n"
        f"{', '.join(key_phrases)}\n\n"

        "### SQL Query:\n"
        f"{sql_query}\n\n"

        "### PropertyKeywords:"
    )

    response = get_groq_completion(client, prompt, temperature=0, max_tokens=700)
    return response.strip()

# ============================================
# Function to Extract User Intent
# ============================================
def get_user_intent_groq(client, query):
    """
    Extract the user intent from the input query using Groq with K-shot prompting.
    """
    prompt = (
        "Analyze the following real estate query and extract the user intent. "
        "Provide the intent as a concise paragraph without any additional text or explanations."
        "If some location or keyword is incomplete,fill them with most appropriate value from the data, for eg: replace redwood with Redwood City\n\n"
        "### Example 1:\n"
        "**Query:** \"Looking for a 3 bedroom house with a big backyard in San Francisco.\"\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards. They likely value outdoor space for activities such as gardening or entertaining.\n\n"
        "### Example 2:\n"
        "**Query:** \"Seeking a 2 bedroom apartment near downtown Seattle with modern amenities.\"\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities. They likely prioritize convenience and contemporary living spaces.\n\n"
        "### Example 3:\n"
        "**Query:** \"2 bed 2 bath in Irvine and 3 bed 2 bath in Redwood under 1600000.\"\n"
        "**User Intent:** The user is looking for both a 2-bedroom, 2-bathroom house in Irvine and a 3-bedroom, 2-bathroom property in Redwood City, with a combined budget under 1,600,000. They seek multiple options within a specific price range.\n\n"
        "### Example 4:\n"
        "**Query:** \"Looking for a 4-bedroom villa in Redwood with a pool and sea view, priced below 2 million.\"\n"
        "**User Intent:** The user desires a luxurious four-bedroom villa in Redwood city that includes a pool and offers a sea view, with a budget below 2 million. They prioritize luxury and scenic views.\n\n"
        "### Example 5:\n"
        "**Query:** \"Searching for 1 bed 1 bath condo in Redwood and 2 bed 2 bath townhouse in Boston under 750000.\"\n"
        "**User Intent:** The user is seeking both a 1-bedroom, 1-bathroom condo in Redwood City and a 2-bedroom, 2-bathroom townhouse in Boston, with a maximum budget of 750,000. They are interested in multiple property types across different cities within a specified price range.\n\n"
        "### Query:\n"
        f"\"{query}\"\n"
        "**User Intent:**"
    )
    response = get_groq_completion(client, prompt)
    return response.strip()

# ============================================
# Function to Extract Traits
# ============================================
def get_traits_groq(client, query, user_intent):
    """
    Extract key traits from the input query using Groq with K-shot prompting.
    Each trait includes a verb phrase and follows a nested bullet structure.
    """
    prompt = (
        "From the following real estate query and user intent, extract the key traits."
        " Provide each trait starting with a verb phrase without any explanations or additional text."
        " Ensure that each trait is concise and relevant to the user's request."
        " Do not split numerical values like prices across multiple lines."
        " If multiple properties are mentioned, list each property separately and combine the budget where applicable."
        " Do not include any preamble, emojis, or phrases like 'Here are the extracted traits:'."
        " If its not explicitly mentioned as house or property, dont add that as a trait"
        "If some location or keyword is incomplete,fill them with most appropriate value from the data, for eg: replace redwood with Redwood City"
        " Your response should only include the traits, exactly as in the examples, and nothing else."
        "\n\n"
        "---\n\n"
        "**Example 1:**\n\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards. They likely value outdoor space for activities such as gardening or entertaining.\n\n"
        "**Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**Traits:**\n"
        " \n"
        "    is a house\n"
        "    has 3 bed, 2 bath\n"
        "    is in San Francisco\n"
        "    has a big backyard\n\n"
        
        "**Example 2:**\n\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities. They likely prioritize convenience and contemporary living spaces.\n\n"
        "**Query:** \"Seeking a 2 bed 1 bath condo in downtown Seattle with modern amenities.\"\n\n"
        "**Traits:**\n"
        " \n"
        "    is a condo\n"
        "    has 2 bed, 1 bath\n"
        "    is in Seattle\n"
        "    has modern amenities\n\n"
        
        "**Example 3:**\n\n"
        "**User Intent:** The user is interested in finding a 2-bedroom, 2-bathroom property in Irvine and a 3-bedroom, 3-bathroom property in San Francisco, with a combined budget of $1,595,000.\n\n"
        "**Query:** \"2 bed 2 bath in Irvine and 3 bed 3 bath in San Francisco both under 1,595,000.\"\n\n"
        "**Traits:**\n"
        " \n"
        "    has 2 bed, 2 bath\n"
        "    is in Irvine\n"
        "    has 3 bed, 3 bath\n"
        "    is in San Francisco\n"
        "    is under $1,595,000.\n\n"
        
        "**Example 4:**\n\n"
        "**User Intent:** The user desires a luxurious four-bedroom villa in Miami that includes a pool and offers a sea view, with a budget below 2 million.\n\n"
        "**Query:** \"Looking for a 4-bedroom villa in Miami with a pool and sea view, priced below 2 million.\"\n\n"
        "**Traits:**\n"
        " \n"
        "    is a villa\n"
        "    has 4 bedrooms\n"
        "    is in Miami\n"
        "    has a pool\n"
        "    has a sea view\n"
        "    is priced below $2,000,000\n\n"
        
        "**Example 5:**\n\n"
        "**User Intent:** The user is seeking both a 1-bedroom, 1-bathroom condo in Redwood City and a 2-bedroom, 2-bathroom townhouse in Boston, with a maximum budget of $750,000.\n\n"
        "**Query:** \"Searching for 1 bed 1 bath condo in Redwood and 2 bed 2 bath townhouse in Boston under 750,000.\"\n\n"
        "**Traits:**\n"
        " \n"
        "    is a condo\n"
        "    has 1 bed, 1 bath\n"
        "    is in Redwood City\n"
        "    is a townhouse\n"
        "    has 2 bed, 2 bath\n"
        "    is in Boston\n"
        "    is under $750,000 (combined)\n\n"
        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Query:**\n"
        f"\"{query}\"\n\n"
        "**Traits:**"
    )
    
    response = get_groq_completion(client, prompt, temperature=0, max_tokens=1500, stop=["\n\n"])
    if response:
        # Split the response into lines and extract the traits
        
        traits_list = [line.rstrip() for line in response.splitlines() if line.strip()]
        return traits_list
    return []





# ============================================
# Function to Extract Key Phrases
# ============================================


def get_key_phrases_groq(client, query, user_intent, traits):
    """
    Extract top key phrases from the input query, user intent, and traits using Groq with K-shot prompting.
    """
    prompt = (
        "From the following real estate query, user intent, and traits, extract the top most relevant key phrases that can be used for search optimization or listing purposes."
        " Provide each key phrase on a separate line without any explanations or additional text."
        " Do not include any preamble, emojis, or phrases like 'Here are the top most relevant key phrases for search optimization or listing purposes:'."
        "If some location or keyword is incomplete,fill them with most appropriate value from the data, for eg: replace redwood with Redwood City"
        " Your response should only include the key phrases, and structure it exactly as in the examples, and add no extra tokens."
        "\n\n"
        "---\n\n"
        "**Example 1:**\n\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards.\n\n"
        "**Traits:**\n"
        "  is a house\n"
        "  has 3 bed, 2 bath\n"
        "  is in San Francisco\n"
        "  has a big backyard\n\n"
        "**Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**Key Phrases:**\n"
        "3 bedroom house\n"
        "big backyard\n"
        "San Francisco real estate\n"
        "spacious home\n"
        "outdoor space\n"
        "family-friendly neighborhood\n"
        "gardening space\n"
        "entertainment area\n"
        "pet-friendly home\n"
        "modern amenities\n\n"
        
        "**Example 2:**\n\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities.\n\n"
        "**Traits:**\n"
        "  is a condo\n"
        "  has 2 bed, 1 bath\n"
        "  is in Seattle\n"
        "  has modern amenities\n\n"
        "**Query:** \"Seeking a 2 bed 1 bath condo in downtown Seattle with modern amenities.\"\n\n"
        "**Key Phrases:**\n"
        "2 bedroom apartment\n"
        "downtown Seattle\n"
        "modern amenities\n"
        "urban living\n"
        "convenient location\n"
        "contemporary design\n"
        "city views\n"
        "public transportation access\n"
        "stylish interiors\n"
        "efficient layout\n\n"
        
        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Traits:**\n"
        f"{traits}\n\n"
        "**Query:**\n"
        f"\"{query}\"\n\n"
        "**Key Phrases:**"
    )
    
    response = get_groq_completion(client, prompt, temperature=0, max_tokens=1000, stop=["\n\n"])
    if response:
        # Split the response into lines and extract key phrases
        key_phrases_list = [line.strip() for line in response.splitlines() if line.strip()]
        return key_phrases_list[:5]
    return []



# ============================================
# Function to Generate User Intent, Traits, and Key Phrases
# ============================================
def extract_information(client, query):
    """
    Extract user intent, traits, and key phrases from the input query.
    """
    user_intent = get_user_intent_groq(client, query)
    traits = get_traits_groq(client, query, user_intent)  # Pass user_intent
    key_phrases = get_key_phrases_groq(client, query, user_intent, traits)  # Pass user_intent
    return user_intent, traits, key_phrases

# ============================================
# Execute the generated SQL query on the Pandas DataFrame using pandasql
# ============================================
def execute_sql_query(sql_query, df):
    """
    Execute the generated SQL query on the Pandas DataFrame using pandasql.
    """
    try:
        # Define 'zillow_data' in the local scope for pandasql
        zillow_data = df

        # Execute the SQL query on the DataFrame
        result = psql.sqldf(sql_query, locals())

        return result
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return None


# Save all the output to a text file
def save_to_txt(file_name, query, user_intent, traits, key_phrases, property_keywords, sql_query, final_output):
    """
    Save the input query, user intent, traits, key phrases, PropertyKeywords, SQL query, and final output to a text file.
    """
    with open(file_name, 'w') as file:
        file.write(f"Input Query: {query}\n\n")
        file.write(f"User Intent: {user_intent}\n\n")
        file.write(f"Traits: {', '.join(traits)}\n\n")
        file.write(f"Key Phrases: {', '.join(key_phrases)}\n\n")
        file.write(f"PropertyKeywords: {property_keywords}\n\n")
        file.write(f"Generated SQL Query: {sql_query}\n\n")
        if final_output is not None and not final_output.empty:
            file.write(f"Final Output from CSV:\n{final_output.to_string(index=False)}\n\n")
        else:
            file.write("Final Output from CSV: No results found.\n\n")
def truncate_text(text, max_length=30):
    """
    Truncate the text to a maximum length and append '...' if truncated.
    
    Parameters:
    - text (str): The original text to truncate.
    - max_length (int): The maximum allowed length of the text.
    
    Returns:
    - str: The truncated text with '...' appended if it was longer than max_length.
    """
    return text if len(text) <= max_length else text[:max_length].rstrip() + '...'



# ============================================
# Initialize session state variables
# ============================================
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'page1'
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'user_intent' not in st.session_state:
        st.session_state.user_intent = ''
    if 'traits' not in st.session_state:
        st.session_state.traits = []
    if 'key_phrases' not in st.session_state:
        st.session_state.key_phrases = []
    if 'property_keywords' not in st.session_state:
        st.session_state.property_keywords = ''
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = ''
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'client' not in st.session_state:
        st.session_state.client = initialize_groq_client()
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    if 'broker_df' not in st.session_state:
        st.session_state.broker_df = load_broker_data()
    if 'selected_broker' not in st.session_state:
        st.session_state.selected_broker = None
    if 'unique_cities' not in st.session_state:
        st.session_state.unique_cities = get_unique_cities(st.session_state.df) if st.session_state.df is not None else []
    
    # Initialize previous searches
    if 'previous_searches' not in st.session_state:
        st.session_state.previous_searches = []
    
    # Initialize selected_zip_code
    if 'selected_zip_code' not in st.session_state:
        st.session_state.selected_zip_code = None


# Navigation functions with callbacks
def go_to_page2():
    st.session_state.page = 'page2'
    st.rerun()

def go_to_page3():
    st.session_state.page = 'page3'
    st.rerun()

def go_to_page1():
    st.session_state.page = 'page1'
    st.rerun()

def go_to_page4():
    st.session_state.page = 'page4'
    st.rerun()



# Main function for Streamlit app
def main():
    st.title("🏠 Real Estate Query App")

    # Initialize session state
    initialize_session_state()

    # Add the sidebar for Previous Searches
    with st.sidebar:
        st.header("🔍 Previous Searches")
        if st.session_state.previous_searches:
            for idx, search in enumerate(st.session_state.previous_searches):
                # Truncate the search query for display
                truncated_query = truncate_text(search['query'], max_length=30)
                
                # Create a button with the search icon and truncated query
                button_label = f"🔍 {truncated_query}"
                # Assign a unique key using the index
                if st.button(button_label, key=f"prev_search_{idx}"):
                    # Load the selected previous search into session state
                    st.session_state.query = search['query']
                    st.session_state.user_intent = search['user_intent']
                    st.session_state.traits = search['traits']
                    st.session_state.key_phrases = search['key_phrases']
                    st.session_state.property_keywords = search['property_keywords']
                    st.session_state.sql_query = search['sql_query']
                    st.session_state.result = search['result']
                    st.session_state.page = 'page2'  # Navigate to the details page
                    st.rerun()
        else:
            st.write("No previous searches found.")


    # Check if client and data are loaded
    if st.session_state.client is None or st.session_state.df is None or st.session_state.broker_df is None:
        st.stop()

    # Page 1: User Input
    if st.session_state.page == 'page1':
        st.header("📥 Page 1: User Input")
        query = st.text_input("Enter your real estate query (e.g., '2 bed 2 bath in Irvine'):", key='user_query')
        if st.button("Submit"):
            if not query.strip():
                st.error("Please enter a valid query.")
            else:
                # Preprocess the query to fix common typos
                preprocessed_query = query
                st.session_state.query = preprocessed_query

                # Extract user intent, traits, and key phrases
                with st.spinner("Extracting information..."):
                    user_intent, traits, key_phrases = extract_information(st.session_state.client, preprocessed_query)
                st.session_state.user_intent = user_intent
                st.session_state.traits = traits
                st.session_state.key_phrases = key_phrases

                # Generate SQL query with retries
                with st.spinner("Generating SQL query..."):
                    sql_query = generate_sql_groq(
                        st.session_state.client, 
                        preprocessed_query, 
                        user_intent, 
                        traits, 
                        key_phrases, 
                        st.session_state.unique_cities, 
                        max_retries=5
                    )
                    st.session_state.sql_query = sql_query

                    if sql_query:
                        # Generate PropertyKeywords
                        with st.spinner("Generating Property Keywords..."):
                            property_keywords = get_property_keywords_groq(
                                st.session_state.client,
                                preprocessed_query,
                                user_intent,
                                traits,
                                key_phrases,
                                sql_query
                            )
                        st.session_state.property_keywords = property_keywords

                        # Execute SQL query
                        with st.spinner("Executing SQL query..."):
                            result = execute_sql_query(sql_query, st.session_state.df)
                        st.session_state.result = result

                        # Save the current search to previous searches
                        new_search = {
                            'query': preprocessed_query,
                            'user_intent': user_intent,
                            'traits': traits,
                            'key_phrases': key_phrases,
                            'property_keywords': property_keywords,
                            'sql_query': sql_query,
                            'result': result
                        }
                        st.session_state.previous_searches.append(new_search)

                        # Optional: Keep only the last 10 searches to limit memory usage
                        if len(st.session_state.previous_searches) > 10:
                            st.session_state.previous_searches.pop(0)

                        # Navigate to Page 2
                        st.session_state.page = 'page2'
                        st.rerun()

                    else:
                        # Handle failure after retries
                        st.error("Could not generate SQL query after multiple attempts. Please try a different query or check your input.")

    # Page 2: Details
    elif st.session_state.page == 'page2':
        st.header("📝 Page 2: Query Details")

        # User Intent
        st.subheader("🔍 User Intent:")
        user_intent = st.session_state.get('user_intent', 'N/A')
        if isinstance(user_intent, str) and user_intent.strip():
            st.write(user_intent)
        else:
            st.write("No user intent information available.")

        # Traits
        st.subheader("📊 Traits:")
        traits = st.session_state.get('traits', [])
        if traits:
            for trait in traits:
                st.write(f"- {trait}")
        else:
            st.write("No traits information available.")

        # Key Phrases
        st.subheader("🗝️ Key Phrases:")
        key_phrases = st.session_state.get('key_phrases', [])
        if key_phrases:
            for phrase in key_phrases[:10]: 
                st.write(f"- {phrase}")
        else:
            st.write("No key phrases available.")

        # **New Section: PropertyKeywords**
        st.subheader("🗝️ Property Keywords:")
        property_keywords = st.session_state.get('property_keywords', '')
        if property_keywords:
            st.write(property_keywords)
        else:
            st.write("No Property Keywords available.")

        # Generated SQL Query
        st.subheader("💻 Generated SQL Query:")
        sql_query = st.session_state.get('sql_query', 'N/A')
        if sql_query:
            # Format SQL using sqlparse
            formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            st.code(formatted_sql, language='sql')
        else:
            st.write("No SQL query generated.")

        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Input"):
                go_to_page1()
        with col2:
            if st.button("➡️ Next to Results"):
                go_to_page3()


                        
    elif st.session_state.page == 'page3':
 
        st.header("📈 Page 3: Query Results")
        
        # Create two columns: one for the toggle switch and one for spacing/content
        # col1, col2 = st.columns([1, 4])  # Adjusted column widths
        

        # Add custom CSS to create a toggle switch
        st.markdown("""
            <style>
            /* The switch - the box around the slider */
            .switch {
                position: relative;
                display: inline-block;
                width: 60px;
                height: 34px;
            }

            /* Hide default HTML checkbox */
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }

            /* The slider */
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
                border-radius: 34px;
            }

            .slider:before {
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }

            input:checked + .slider {
                background-color: #2196F3;
            }

            input:checked + .slider:before {
                transform: translateX(26px);
            }
            </style>
            """, unsafe_allow_html=True)

        # Create the toggle button and a label for it
        st.markdown("""
            <label for="toggle-switch">Notify Property Alerts: </label>
            <label class="switch">
                <input type="checkbox" id="property-alert-toggle">
                <span class="slider"></span>
            </label>
            <script>
                const toggle = document.getElementById("property-alert-toggle");
                toggle.addEventListener("change", function() {
                    if (toggle.checked) {
                        alert("🔔 Property alerts are enabled.");
                    } else {
                        alert("🔕 Property alerts are disabled.");
                    }
                });
            </script>
        """, unsafe_allow_html=True)
    
        result = st.session_state.get('result')

        if result is not None and not result.empty:
            st.write(f"### 🏡 {len(result)} Properties Found:")

            st.write("### 🏡 Properties:")
            result = result.reset_index(drop=True)
            
            # Ensure 'beds' and 'baths' are integers
            result['beds'] = pd.to_numeric(result['beds'], errors='coerce').fillna(0).astype(int)
            result['baths'] = pd.to_numeric(result['baths'], errors='coerce').fillna(0).astype(int)
            
            # Initialize dynamic_columns as an empty list
            dynamic_columns = []
            
            # Extract traits and key phrases from session state
            traits = st.session_state.get('traits', [])
            key_phrases = st.session_state.get('key_phrases', [])
            
            # ============================================
            # 1. Extract Bed-Bath Combinations
            # ============================================
            bed_bath_conditions = extract_bed_bath_combinations(st.session_state.sql_query)
            # st.write("Bed-Bath Conditions Extracted:", bed_bath_conditions)
            
            # Create bed-bath combination strings (e.g., "2 bed 2 bath")
            bed_bath_columns = [f"{condition['beds']} bed {condition['baths']} bath" for condition in bed_bath_conditions]
            
            # Append bed_bath_columns to dynamic_columns
            dynamic_columns += bed_bath_columns
            
            # ============================================
            # 2. Extract Additional Features
            # ============================================
            additional_features = extract_additional_features(traits, key_phrases, possible_features, feature_synonyms)
            # st.write("Additional Features Extracted:", additional_features)
            
            # Append additional_features to dynamic_columns
            dynamic_columns += additional_features
            
            # ============================================
            # 3. Create Dynamic Columns for Bed-Bath
            # ============================================
            for bed_bath_str in bed_bath_columns:
                if bed_bath_str not in result.columns:
                    # Extract bed and bath numbers
                    bed_num = int(bed_bath_str.split()[0])
                    bath_num = int(bed_bath_str.split()[2])
                    
                    # Create the dynamic column with 🟢 or ⚪
                    result[bed_bath_str] = result.apply(
                        lambda row: "🟢" if row.get('beds', 0) == bed_num and row.get('baths', 0) == bath_num else "⚪",
                        axis=1
                    )
                    print(f"Created column: {bed_bath_str}")  # Debugging statement
            
            # ============================================
            # 4. Handle Additional Features
            # ============================================
            # Create 'first_school_rating' column if 'good school nearby' or 'excellent school nearby' is present
            if 'good school nearby' in dynamic_columns or 'excellent school nearby' in dynamic_columns:
                result['first_school_rating'] = result['school_ratings'].apply(extract_first_school_rating)
                # st.write("First School Rating:", result['first_school_rating'].head())  # Debugging statement

            for column in dynamic_columns:
                if column == 'good school nearby':
                    # Use the first_school_rating to determine
                    result[column] = result['first_school_rating'].apply(lambda x: "🟢" if pd.notna(x) and x >= 7 else "⚪")
                elif column == 'excellent school nearby':
                    # Use the first_school_rating to determine
                    result[column] = result['first_school_rating'].apply(lambda x: "🟢" if pd.notna(x) and x >= 8 else "⚪")
                elif column in possible_features and column not in ['good school nearby', 'excellent school nearby']:
                    # Features like 'pool', 'restaurant'
                    result[column] = result['neighborhood_desc'].apply(lambda desc: "🟢" if check_feature_in_description(desc, column) else "⚪")
                else:
                    # Bed-bath columns have already been handled
                    pass
            
            # ============================================
            # 5. Prepare Display DataFrame
            # ============================================
            # Define the columns to display: Property Link, Price, City, Dynamic Columns, Description, Broker, Receive Realtor Proposal
            display_columns = ['Property Link', 'Price', 'City'] + dynamic_columns + ['Description', 'Broker', 'Receive Realtor Proposal']

            # Prepare display DataFrame
            display_df = pd.DataFrame()

            # Property Link
            display_df['Property Link'] = result['listingurl'].apply(
                lambda url: f"[View Property]({url})" if pd.notna(url) else "N/A"
            )

            # Price
            display_df['Price'] = result['price'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )

            # City (Fixed Column)
            display_df['City'] = result['city'].apply(
                lambda x: x.title() if isinstance(x, str) else "N/A"
            )

            # Dynamic Columns (bed-bath and others)
            for column in dynamic_columns:
                display_df[column] = result[column]
            
            # Description
            display_df['Description'] = result['neighborhood_desc'].apply(
                lambda desc: f"<div style='height: 100px; overflow: auto;'>{desc}</div>" if isinstance(desc, str) else "N/A"
            )

            # Broker (Clickable Link to Page 4)
            display_df['Broker'] = result['zip_code'].apply(
                lambda zip_code: "Click to view brokers" if pd.notna(zip_code) else "N/A"
            )

            # Receive Realtor Proposal (Button Placeholder)
            display_df['Receive Realtor Proposal'] = "Button Placeholder"

            # ============================================
            # 6. Display the DataFrame in Streamlit
            # ============================================
            # Display the header row with column names
            with st.container():
                cols = st.columns(len(display_columns))
                for col, header in zip(cols, display_columns):
                    col.markdown(f"**{header}**")

            # Display each property row
            for index, row in display_df.iterrows():
                with st.container():
                    cols = st.columns(len(display_columns))
                    # Property Link
                    cols[0].markdown(row['Property Link'], unsafe_allow_html=True)
                    # Price
                    cols[1].markdown(row['Price'], unsafe_allow_html=True)
                    # City
                    cols[2].markdown(row['City'], unsafe_allow_html=True)
                    # Dynamic Columns (bed-bath and others)
                    for i, column in enumerate(dynamic_columns, start=3):
                        cols[i].markdown(row[column], unsafe_allow_html=True)
                    # Description
                    cols[-3].markdown(row['Description'], unsafe_allow_html=True)
                    # Broker Link
                    with cols[-2]:
                        if st.button("🔗 Click to view brokers", key=f"view_brokers_{index}"):
                            selected_zip = result.loc[index, 'zip_code']
                            st.session_state.selected_zip_code = selected_zip
                            st.session_state.page = 'page4'
                            st.rerun()
                    # Receive Realtor Proposal Button
                    with cols[-1]:
                        if st.button("📨 Receive Realtor Proposal", key=f"realtor_proposal_{index}"):
                            # Define the button's action here
                            st.success(f"Realtor proposal for property in {row['City']} has been sent!")
                            # You can also add more complex logic, such as sending an email or saving to a database

            # Optionally, offer to download the results as CSV
            csv = result.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='query_results.csv',
                mime='text/csv',
            )
        else:
            st.write("### ❌ No results found for the given query.")

        # Display the number of properties found
        if result is not None:
            st.write(f"### 🏠 Total Properties Found: {len(result)}")
        else:
            st.write("### 🏠 Total Properties Found: 0")

        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Details", key="back_to_details"):
                go_to_page2()
        with col2:
            if st.button("💾 Save Results to Text File", key="save_results"):
                save_to_txt(
                    file_name="query_output.txt",
                    query=st.session_state.get('query', ''),
                    user_intent=st.session_state.get('user_intent', ''),
                    traits=st.session_state.get('traits', []),
                    key_phrases=st.session_state.get('key_phrases', []),
                    property_keywords=st.session_state.get('property_keywords', ''),
                    sql_query=st.session_state.get('sql_query', ''),
                    final_output=result
                )
                st.success("Results saved to 'query_output.txt'.")
            
    # Page 4: Broker Details
    # Page 4: Broker Details
    elif st.session_state.page == 'page4':
        st.header("📄 Page 4: Broker Details")

        selected_zip_code = st.session_state.get('selected_zip_code', None)

        if selected_zip_code:
            broker_df = st.session_state.get('broker_df', None)
            if broker_df is not None:
                # Filter brokers by the selected zip code
                filtered_brokers = broker_df[broker_df['zip_code'] == selected_zip_code]
                if not filtered_brokers.empty:
                    st.subheader(f"🏢 Brokers in Zip Code: {selected_zip_code}")
                    
                    # Select and rename columns for better readability
                    display_brokers = filtered_brokers.rename(columns={
                        'broker': 'Broker Name',
                        'zip_code': 'Zip Code',
                        'city': 'City',
                        'state': 'State',
                        'reviews': 'Reviews',
                        'recent_homes_sold': 'Recent Homes Sold',
                        'negotiations_done': 'Negotiations Done',
                        'years_of_experience': 'Years of Experience',
                        'rating': 'Rating'
                    })[['Broker Name', 'City', 'State', 'Zip Code', 'Reviews', 'Recent Homes Sold', 'Negotiations Done', 'Years of Experience', 'Rating']]

                    # Display the brokers in a table
                    st.dataframe(display_brokers.style.format({
                        'Price': "${:,.2f}",
                        'Rating': "{:.2f}"
                    }))
                else:
                    st.write(f"No brokers found for zip code: {selected_zip_code}")
            else:
                st.write("Broker data is not available.")
        else:
            st.write("No zip code selected.")

        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Results"):
                go_to_page3()
        with col2:
            if st.button("🏠 Back to Input"):
                go_to_page1()

    # Default case: reset to Page 1
    else:
        st.session_state.page = 'page1'
        st.rerun()


if __name__ == '__main__':
    main()

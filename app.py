import os
import google.generativeai as genai
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import logging
import streamlit as st
import google.generativeai as genai

# Configure logging to include DEBUG level and output to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def load_api_key():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logging.error("GEMINI_API_KEY not found in environment variables.")
        raise EnvironmentError("GEMINI_API_KEY not set.")
    return api_key

def configure_genai(api_key):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        logging.info("Successfully configured Gemini model.")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        raise

def load_csv(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"Successfully loaded CSV file: {csv_file_path}")
        logging.debug(f"DataFrame head:\n{df.head()}")
        return df
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        raise

def clean_dataframe(df):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
    df['days_on_market'] = pd.to_numeric(df['days_on_market'], errors='coerce')
    return df

def prepare_k_shot_prompt(df, num_examples=3):
    """Prepare a few-shot prompt with examples from the DataFrame."""
    k_shot_examples = []
    for i in range(min(num_examples, len(df))):
        example = {
            "price": int(df.iloc[i]['price']) if not pd.isna(df.iloc[i]['price']) else None,
            "address": str(df.iloc[i]['address']) if not pd.isna(df.iloc[i]['address']) else "N/A",
            "beds": int(df.iloc[i]['beds']) if not pd.isna(df.iloc[i]['beds']) else None,
            "baths": int(df.iloc[i]['baths']) if not pd.isna(df.iloc[i]['baths']) else None,
            "area": float(df.iloc[i]['area']) if not pd.isna(df.iloc[i]['area']) else None,
            "year_built": int(df.iloc[i]['year_built']) if not pd.isna(df.iloc[i]['year_built']) else None,
            "days_on_market": int(df.iloc[i]['days_on_market']) if not pd.isna(df.iloc[i]['days_on_market']) else None,
            "neighborhood_desc": str(df.iloc[i]['neighborhood_desc']) if not pd.isna(df.iloc[i]['neighborhood_desc']) else "N/A"
        }
        k_shot_examples.append(json.dumps(example))
    return "\n".join(k_shot_examples)

def generate_query_json(model, k_shot_prompt, input_query):
    prompt = (
    "Given the following examples, return similar JSON (study only the column names, don't take contents from it) for the query provided. The structure should match the examples.\n\n"
    "Make sure you capture the following values if the user specifies them: search_criteria, address, listingUrl, price, beds, baths, area, coord, zestimate, priceReduction, brokerName, crawl_url_result, days_on_market, page_views, listing_agent, year_built, hoa_fees, property_tax, school_ratings, neighborhood_desc. "
    "Take all values as string except price, beds, baths, area, coord, zestimate, priceReduction, days_on_market, page_views, year_built, hoa_fees, property_tax, school_ratings. For price, beds, baths, area, coord, zestimate, priceReduction, days_on_market, page_views, year_built, hoa_fees, property_tax, school_ratings, give output as int. "
    "In the output, make sure it does not give content outside the query. If details are not provided, return null for numeric fields and \"N/A\" for string fields.\n\n"
    "Return the JSON response only within triple backticks and specify 'json' after the backticks.\n\n"
    "Now return a result for the input query: "
    f"{input_query}"
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=1024,
                temperature=0.7
            ),
        )
        if response and response.candidates:
            generated_content = response.candidates[0].content.parts[0].text
            logging.info("Successfully generated content from Gemini model.")
            return generated_content
        else:
            logging.error("No valid response from the Gemini model.")
            return None
    except Exception as e:
        logging.error(f"Error during content generation: {e}")
        return None

def clean_generated_json(generated_content):
    """Clean the generated JSON content to ensure it can be parsed."""
    json_block = re.search(r'```json\s*(\{.*\})\s*```', generated_content, re.DOTALL)
    if json_block:
        cleaned_content = json_block.group(1)
    else:
        json_block = re.search(r'\{.*\}', generated_content, re.DOTALL)
        if json_block:
            cleaned_content = json_block.group()
        else:
            cleaned_content = re.sub(r"```json|```", "", generated_content).strip()
    
    numeric_fields = [
        'price', 'beds', 'baths', 'area', 'coord', 'zestimate',
        'priceReduction', 'days_on_market', 'page_views',
        'year_built', 'hoa_fees', 'property_tax', 'school_ratings'
    ]

    for field in numeric_fields:
        pattern = rf'("{field}": )N/A'
        replacement = rf'\1null'
        cleaned_content = re.sub(pattern, replacement, cleaned_content)
    
    cleaned_content = re.sub(r',\s*([}\]])', r'\1', cleaned_content)
    
    return cleaned_content

def parse_json(cleaned_content):
    """Parse the cleaned JSON content."""
    try:
        query_data = json.loads(cleaned_content)
        if isinstance(query_data, list) and len(query_data) > 0:
            query = query_data[0]
        elif isinstance(query_data, dict):
            query = query_data
        else:
            query = None
        
        if query:
            query_text = (
                f"Price: {query.get('price', 'N/A')}, Address: {query.get('address', 'N/A')}, "
                f"Beds: {query.get('beds', 'N/A')}, Baths: {query.get('baths', 'N/A')}, "
                f"Area: {query.get('area', 'N/A')}, Neighborhood: {query.get('neighborhood_desc', 'N/A')}"
            )
            return query
        else:
            return None
    except json.JSONDecodeError as e:
        return None

def filter_initial(df, query):
    try:
        beds = query.get('beds', 0)
        baths = query.get('baths', 0)
        price = query.get('price', np.inf)
        
        if price is None or pd.isna(price):
            price = np.inf
        
        if beds is None:
            beds = 0
        if baths is None:
            baths = 0
        
        df_filtered = df.copy()
        df_filtered['price'] = df_filtered['price'].fillna(np.inf)
        df_filtered['beds'] = df_filtered['beds'].fillna(0)
        df_filtered['baths'] = df_filtered['baths'].fillna(0)
        
        filtered_df = df_filtered.loc[
            (df_filtered['beds'] == beds) &
            (df_filtered['baths'] == baths) &
            (df_filtered['price'] <= price)
        ]
        return filtered_df
    except TypeError as e:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def compute_text_embeddings(df, embedding_model):
    """Compute embeddings for the textual data in the DataFrame."""
    # Combine 'address' and 'neighborhood_desc' into a single text field
    df['text'] = df['address'].astype(str) + ' ' + df['neighborhood_desc'].astype(str)
    # Compute embeddings
    texts = df['text'].tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    df['embedding'] = list(embeddings)
    return df

def filter_with_embeddings(df, input_query, embedding_model, top_n=10):
    """Filter the DataFrame using embeddings."""
    # Compute the embedding of the user's original query text
    query_embedding = embedding_model.encode([input_query])[0]

    # Compute cosine similarities
    embeddings = np.array(df['embedding'].tolist())
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Add similarities to dataframe
    df['similarity'] = similarities

    # Sort by similarity
    df = df.sort_values(by='similarity', ascending=False)

    # Return top_n results
    return df.head(top_n)

def main():
    # Initialize session state for 'query' and 'input_query' if they don't exist
    if 'query' not in st.session_state:
        st.session_state['query'] = None
    if 'input_query' not in st.session_state:
        st.session_state['input_query'] = ""

    GEMINI_API_KEY = load_api_key()
    model = configure_genai(GEMINI_API_KEY)

    csv_file_path = 'Zillow listing data.csv'
    df = load_csv(csv_file_path)
    df = clean_dataframe(df)

    # Load the embedding model
    model_name = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)

    # Compute embeddings
    df = compute_text_embeddings(df, embedding_model)

    query_params = st.query_params
    page = query_params.get("page", ["1"])[0]

    if page == "1":
        st.title("Real Estate Query Input")
        input_query = st.text_area("Enter your real estate query:", "")
        if st.button("Submit Query"):
            k_shot_prompt = prepare_k_shot_prompt(df)
            generated_content = generate_query_json(model, k_shot_prompt, input_query)
            if generated_content:
                cleaned_content = clean_generated_json(generated_content)
                query = parse_json(cleaned_content)
                st.session_state["query"] = query
                st.session_state["input_query"] = input_query  # Store the user's original input query
                # Only rerun if the page parameter is changing
                if st.query_params.get("page", ["1"])[0] != "2":
                    st.query_params = {"page": "2"}
                    st.rerun()
            else:
                st.error("Failed to generate query. Please try again.")


    elif page == "2":
        st.title("Extracted Information")
        query = st.session_state.get("query", None)
        if query:
            st.json(query)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back"):
                    if st.query_params.get("page", ["2"])[0] != "1":
                        st.query_params = {"page": "1"}
                        st.rerun()
            with col2:
                if st.button("See Results"):
                    if st.query_params.get("page", ["2"])[0] != "3":
                        st.query_params = {"page": "3"}
                        st.rerun()
        else:
            st.error("No query data available. Please go back and submit a query.")
            if st.button("Back to Query Input"):
                if st.query_params.get("page", ["2"])[0] != "1":
                    st.query_params = {"page": "1"}
                    st.rerun()

    elif page == "3":
        st.title("Final Results")
        query = st.session_state.get("query", None)
        input_query = st.session_state.get("input_query", "")
        if query and input_query:
            filtered_df = filter_initial(df, query)
            if not filtered_df.empty:
                filtered_df = filtered_df.reset_index(drop=True)
                filtered_df = filter_with_embeddings(filtered_df, input_query, embedding_model)
                if not filtered_df.empty:
                    st.write(filtered_df)
                else:
                    st.warning("No matching properties found after embedding filtering.")
            else:
                st.warning("No matching properties found.")
            if st.button("Back"):
                if st.query_params.get("page", ["3"])[0] != "1":
                    st.query_params = {"page": "1"}
                    st.rerun()
        else:
            st.error("No query data available. Please go back and submit a query.")
            if st.button("Back to Query Input"):
                if st.query_params.get("page", ["3"])[0] != "1":
                    st.query_params = {"page": "1"}
                    st.rerun()

    else:
        st.error("Invalid page parameter.")
        if st.button("Back to Query Input"):
            if st.query_params.get("page", ["1"])[0] != "1":
                st.query_params = {"page": "1"}
                st.rerun()

if __name__ == "__main__":
    main()


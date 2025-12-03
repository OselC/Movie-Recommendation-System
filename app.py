import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import pickle
import pandas as pd
import os
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_desc(overview):
    # overview is a valid string
    if not isinstance(overview, str):
        return ""

    # remove extra spaces
    overview = ' '.join(overview.split())

    # tokenize word
    words = word_tokenize(overview)
    # ada uppercase kita ganti jadi lowercase
    words = [w.lower() for w in words if w.isalpha() and w not in stop_words]

    # apply stemming
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)

@st.cache_resource
def load_dataset():
    try:
        df = pd.read_csv("Cleaned_imdb_dataset.csv")
        return df

    except FileNotFoundError as e:
        st.error(f"Asset not found: {e}. Please ensure the dataset is available.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during asset loading: {e}")
        st.stop()

def load_train_model(df, model_file='model.pickle'):

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            tfidf, mlb, combined_features = pickle.load(f)
            print("Model loaded from model.pickle")
        return tfidf, mlb, combined_features
    
    overview = df['Overview']

    # preprocess overview column
    overview = overview.apply(preprocess_desc)

    # Split genre by comma and strip spaces
    df["genre_list"] = df["Genre"].apply(lambda x: [g.strip() for g in x.split(",")])

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genre_list"])

    tfidf = TfidfVectorizer(stop_words='english')
    overview_matrix = tfidf.fit_transform(overview)

    combined_features = hstack([overview_matrix * 0.7, genre_matrix * 0.3])
    
    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump((tfidf, mlb, combined_features), f)
        print("Model created and saved successfully!")

    return tfidf, mlb, combined_features

def movie_recommendation(query_input, tfidf, mlb, combined_features, df, top_k=5):
    movie_titles = df['Series_Title'].values
    imdb_ratings = df['IMDB_Rating'].values
    genres = df['Genre'].values
    overview = df['Overview'].values

    # preprocess
    query_preprocess = preprocess_desc(query_input)

    # vectorize
    query_overview_vector = tfidf.transform([query_preprocess])

    # Get the number of genre features
    num_genre_features = len(mlb.classes_)
    query_genre_vector = np.zeros((1, num_genre_features))

    query_vector = hstack([query_overview_vector, query_genre_vector])

    # calculate cosine similarity
    cosine_sim = cosine_similarity(query_vector, combined_features)[0]

    # sort indices by similarity (descending)
    sorted_indices = np.argsort(cosine_sim)[::-1]

    # take top K movies
    top_indices = sorted_indices[:top_k]

    # Extract recommendation data
    top_movies = movie_titles[top_indices]
    top_ratings = imdb_ratings[top_indices]
    top_genres = genres[top_indices]
    top_overviews = overview[top_indices]
    top_scores = cosine_sim[top_indices]

    # Create a DataFrame for output
    recommendation_df = pd.DataFrame({
        'Series_Title': top_movies,
        'Genre': top_genres,
        'IMDB_Rating': top_ratings,
        'Overview' : top_overviews,
        'Similarity_Score': top_scores
    })
    
    return recommendation_df

df = load_dataset()
tfidf, mlb, combined_features = load_train_model(df)

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🎬 Movie Recommendation System")
st.markdown("Enter a short description of the type of movie you want to watch.")

# --- User Input ---
query_input = st.text_area(
    "What kind of movie are you looking for?",
    placeholder="e.g., A gripping sci-fi film about time travel and destiny.",
    height=100
)

# --- Recommendation Generation ---
if st.button("Get Recommendations", use_container_width=True):
    if query_input:
        with st.spinner('Searching for the best matches...'):
            # Get recommendations
            results_df = movie_recommendation(query_input, tfidf, mlb, combined_features, df)
            
            # Display results
            st.subheader(f"Top {len(results_df)} Recommendations")
            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True
            )
            
            st.markdown(f"*(Total movies indexed: {len(df)})*")
    else:
        st.warning("Please enter a description to get recommendations.")
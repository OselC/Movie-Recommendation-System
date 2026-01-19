# Movie-Recommendation-System
A **content-based movie recommendation system** that processes user prompts and leverages TF-IDF/embedding-based similarity to recommend movies based on a user’s textual description by analyzing features such as movie plot descriptions and genres.

Dataset: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows 

Streamlit App: https://movie-recommendation-system-namd3hkt3hesw9bb7seygf.streamlit.app/

---

## 🚀 Features
- Text-based movie search (describe what you want to watch)
- NLP preprocessing (tokenization, stopword removal, stemming)
- TF-IDF vectorization for movie overviews
- Genre encoding using MultiLabelBinarizer
- Weighted feature combination (70% overview, 30% genre)
- Cosine similarity for recommendation ranking
- Interactive web UI with Streamlit
- Cached model loading for fast performance

## 🧠 How It Works
1. Movie descriptions are cleaned and preprocessed.
2. Overviews are converted into TF-IDF vectors.
3. Genres are transformed into multi-label binary vectors.
4. Both features are combined into a single vector space.
5. User input is vectorized and compared using cosine similarity.
6. The top similar movies are returned with:
   - Title
   - Genre
   - IMDb Rating
   - Overview
   - Similarity Score

## 🛠 Tech Stack
- Python
- Streamlit  
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  
- SciPy  

## How to run the app in Streamlit
1. Go to file directory and type:
```bash
cmd
code .
```

2. Go to terminal and type this command
```bash
pip install -r requirements.txt
streamlit run app.py
```

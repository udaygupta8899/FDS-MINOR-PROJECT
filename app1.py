import streamlit as st
import requests
import pandas as pd
from random import sample
import os
from typing import List, Dict

class MovieRecommender:
    def __init__(self, api_key: str):
        """Initialize the MovieRecommender with TMDB API key"""
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.genres = self._get_genres()
        
    def _get_genres(self) -> Dict[str, int]:
        """Fetch all available genres from TMDB"""
        url = f"{self.base_url}/genre/movie/list"
        params = {"api_key": self.api_key}
        
        response = requests.get(url, params=params)
        genres = {genre["name"]: genre["id"] for genre in response.json()["genres"]}
        return genres
    
    def get_movies_by_genres(self, genre_ids: List[int], page: int = 1) -> List[Dict]:
        """Fetch movies that match both selected genres"""
        url = f"{self.base_url}/discover/movie"
        params = {
            "api_key": self.api_key,
            "with_genres": ",".join(map(str, genre_ids)),
            "page": page,
            "sort_by": "popularity.desc"
        }
        
        response = requests.get(url, params=params)
        return response.json()["results"]
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Fetch detailed information for a specific movie"""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {"api_key": self.api_key}
        
        response = requests.get(url, params=params)
        return response.json()

def main():
    st.set_page_config(page_title="Movie Genre Fusion Recommender", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .movie-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #1e2130;
            margin-bottom: 1rem;
        }
        .movie-title {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .movie-info {
            color: #c0c0c0;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🎬 Movie Genre Fusion Recommender")
    st.markdown("""
        Discover movies that combine your favorite genres! Select two genres below 
        to find movies that perfectly blend both styles.
    """)
    
    # Initialize MovieRecommender with your TMDB API key
    api_key = st.secrets["TMDB_API_KEY"]  # Store your API key in Streamlit secrets
    recommender = MovieRecommender(api_key)
    
    # Genre selection
    col1, col2 = st.columns(2)
    with col1:
        genre1 = st.selectbox(
            "Select your first genre",
            options=sorted(recommender.genres.keys()),
            key="genre1"
        )
    
    with col2:
        genre2 = st.selectbox(
            "Select your second genre",
            options=[g for g in sorted(recommender.genres.keys()) if g != genre1],
            key="genre2"
        )
    
    if st.button("Find Movies"):
        with st.spinner("Searching for your perfect movies..."):
            # Get genre IDs
            genre_ids = [recommender.genres[genre1], recommender.genres[genre2]]
            
            # Fetch movies
            movies = recommender.get_movies_by_genres(genre_ids)
            
            if movies:
                st.subheader(f"Movies combining {genre1} and {genre2}")
                
                # Display movies in a grid
                cols = st.columns(3)
                for idx, movie in enumerate(movies[:9]):  # Show top 9 movies
                    with cols[idx % 3]:
                        # Create movie card
                        st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">{movie['title']}</div>
                                <div class="movie-info">
                                    Rating: ⭐ {movie['vote_average']}/10<br>
                                    Release: {movie['release_date']}<br>
                                </div>
                                <div class="movie-info" style="margin-top: 0.5rem;">
                                    {movie['overview'][:150]}...
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add poster if available
                        if movie['poster_path']:
                            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                            st.image(poster_url, use_column_width=True)
            else:
                st.warning("No movies found combining these genres. Try different genres!")
    
    # Add footer with credits
    st.markdown("""
        ---
        Data provided by [The Movie Database (TMDB)](https://www.themoviedb.org).
        This product uses the TMDB API but is not endorsed or certified by TMDB.
    """)

if __name__ == "__main__":
    main()

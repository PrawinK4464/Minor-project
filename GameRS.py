import streamlit as st
import pandas as pd
import pickle
import requests
import os

# Streamlit layout
st.set_page_config(page_title="🎮 GameRS 🎮 | Looking for your next gaming adventure? Dive into our treasure trove of game recommendations tailored just for you!", page_icon="img/logo.png", layout="wide")

# Load the CSS file
with open('static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the pickled DataFrame and cosine similarity matrix
with open('data/games_data5000.pkl', 'rb') as f:
    games_df = pickle.load(f)

with open('data/cosine_sim5000.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# RAWG API Configuration
API_KEY = 'e2d29969767343249f9a1bd62880b056'
BASE_URL = 'https://api.rawg.io/api/games'

# Recommendation function
def recommend_games(game_title, games_df=games_df, alpha=0.5, beta=0.3, gamma=0.2):
    if game_title not in games_df['name'].values:
        return None  # Return None if the game is not found
    idx = games_df[games_df['name'] == game_title].index[0]
    selected_game_row = games_df.iloc[idx]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    game_indices = [i[0] for i in sim_scores]
    recommended_games = games_df.iloc[game_indices]

    # Calculate predicted enjoyment scores using the updated formula
    recommended_games['predicted_enjoyment'] = recommended_games.apply(
        lambda row: calculate_predicted_enjoyment(row, selected_game_row, games_df, alpha, beta, gamma), axis=1
    )

    # Rank recommendations based on predicted enjoyment
    recommended_games = recommended_games.sort_values(by='predicted_enjoyment', ascending=False)

    return recommended_games


def calculate_predicted_enjoyment(game_row, selected_game_row, games_df, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Calculate USIG based on a weighted formula using rating, genre similarity, and platform similarity.
    
    Parameters:
    - game_row: The row of the game to calculate the USIG for.
    - selected_game_row: The row of the selected game (reference game).
    - games_df: DataFrame of all games.
    - alpha, beta, gamma: Weights assigned to the factors rating, genre similarity, and platform similarity.
    
    Returns:
    - USIG: User Satisfaction Index for Games (0 to 1 scale)
    """
    # Normalized rating (rating should be between 0 and 5, so normalize by dividing by 5)
    normalized_rating = game_row['rating'] / 5 if not pd.isnull(game_row['rating']) else 0

    # Genre similarity: based on how many genres overlap
    selected_game_genres = set(selected_game_row['genres']) if isinstance(selected_game_row['genres'], list) else set()
    game_genres = set(game_row['genres']) if isinstance(game_row['genres'], list) else set()
    genre_similarity = len(selected_game_genres.intersection(game_genres)) / len(selected_game_genres.union(game_genres)) if selected_game_genres and game_genres else 0

    # Platform similarity: based on platform overlap
    selected_game_platforms = set(selected_game_row['platforms']) if isinstance(selected_game_row['platforms'], list) else set()
    game_platforms = set(game_row['platforms']) if isinstance(game_row['platforms'], list) else set()
    platform_similarity = len(selected_game_platforms.intersection(game_platforms)) / len(selected_game_platforms.union(game_platforms)) if selected_game_platforms and game_platforms else 0

    # Weighted sum of factors
    usig_score = (alpha * normalized_rating + beta * genre_similarity + gamma * platform_similarity) / (alpha + beta + gamma)
    
    return round(usig_score, 2)  # Round to 2 decimal places


# Fetch game details from RAWG API
def fetch_game_details(game_id):
    url = f'{BASE_URL}/{game_id}?key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Uh-oh! Something went wrong while fetching game details.")
        return None

# Fetch game trailers
def fetch_game_trailers(game_id):
    url = f'{BASE_URL}/{game_id}/movies?key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        trailers_data = response.json()
        return trailers_data
    else:
        st.error("Oh no! We couldn't find any trailers for this game.")
        return None

st.title("🎮 GameRS 🎮")
st.subheader("Looking for your next gaming adventure? Dive into our treasure trove of game recommendations tailored just for you!")

# Sidebar for user input
st.sidebar.header("🔍 Search for Your Favorite Game")
selected_game = st.sidebar.selectbox("Pick a game from our collection:", games_df['name'])

# Fetch details of the selected game
if selected_game:
    game_details = fetch_game_details(games_df[games_df['name'] == selected_game]['id'].values[0])
    if game_details:
        st.sidebar.subheader(f"🎮 Game Details for **{selected_game}**")
        st.sidebar.image(game_details.get('background_image', ''), width=200)

        # Minimizable description
        full_description = game_details.get('description_raw', 'No description available')
        with st.sidebar.expander("🔍 Show Description", expanded=False):
            st.write(full_description)

        # Display the game details in the sidebar
        platforms = ', '.join([platform['platform']['name'] for platform in game_details.get('platforms', [])])
        st.sidebar.write(f"<span class='platform'>**Platforms:**</span> {platforms if platforms else 'Not available'}", unsafe_allow_html=True)
        st.sidebar.write(f"<span class='metacritic'>**Metacritic Score:**</span> {game_details.get('metacritic', 'N/A')}", unsafe_allow_html=True)
        st.sidebar.write(f"<span class='genre'>**Genres:**</span> {', '.join([genre['name'] for genre in game_details.get('genres', [])])}", unsafe_allow_html=True)

        # Tags as minimizable
        if 'tags' in game_details and game_details['tags']:
            tags = [tag['name'] for tag in game_details['tags']]
            with st.sidebar.expander("🏷️ Show Tags", expanded=False):
                st.write(f"<span class='tag'>Tags:</span> {', '.join(tags)}", unsafe_allow_html=True)
        else:
            st.sidebar.write("<span class='tag'>Tags:</span> Not available", unsafe_allow_html=True)

        # Check for ESRB rating safely
        esrb_rating = game_details.get('esrb_rating')
        if esrb_rating:
            st.sidebar.write(f"<span class='esrb'>**ESRB Rating:**</span> {esrb_rating.get('name', 'N/A')}", unsafe_allow_html=True)
        else:
            st.sidebar.write("<span class='esrb'>**ESRB Rating:**</span> Not available", unsafe_allow_html=True)

        st.sidebar.write(f'<a class="sidebar-more-info-button" href="https://rawg.io/games/{game_details.get("slug", "")}" target="_blank">📝 More Info</a>', unsafe_allow_html=True)

        # Available on
        if 'stores' in game_details and game_details['stores']:
            stores = [store['store']['name'] for store in game_details['stores']]
            st.sidebar.write(f"<span class='platform'>**Available on:**</span> {', '.join(stores)}", unsafe_allow_html=True)
        else:
            st.sidebar.write("<span class='platform'>**Available on:**</span> Not available", unsafe_allow_html=True)

        # Fetch trailers
        trailers = fetch_game_trailers(games_df[games_df['name'] == selected_game]['id'].values[0])
        if trailers and trailers.get('results'):
            trailer_url = trailers['results'][0]['data']['max']
            if trailer_url:
                with st.sidebar.expander("📽️ Watch Trailer", expanded=False):
                    st.video(trailer_url)
            else:
                st.sidebar.write("**Trailer not available.**")
        else:
            st.sidebar.write("**No trailers found for this game.**")
    else:
        st.sidebar.write("**Game details not available.**")


# Show recommendations
if st.sidebar.button("🌟 Get Recommendations"):
    with st.spinner("Loading recommendations..."):
        recommendations = recommend_games(selected_game)
    if recommendations is not None:
        st.subheader(f"✨ Recommendations for **{selected_game}**:")
        for i, row in recommendations.iterrows():
            st.write(f"### {row['name']}")
            st.image(row['background_image'], width=200)
            st.write(f"<span class='esrb'>**USIG:**</span> {row['predicted_enjoyment']} <span style='font-size: 0.8em; color: gray;'>(User Satisfaction Index for Games: Calculated based on Rating, Genres, and Platforms)</span>", unsafe_allow_html=True)
            st.write(f'<a class="more-info-button" href="https://rawg.io/games/{row["slug"]}" target="_blank">📝 More Info</a>', unsafe_allow_html=True)

            # Fetch additional details from RAWG API
            game_details = fetch_game_details(row['id'])
            if game_details:
                full_description = game_details.get('description_raw', 'No description available')  # Full description
                
                # Use expander for the full description
                with st.expander("🔍 Show Description", expanded=False):
                    st.write(full_description)

                st.write(f"<span class='metacritic'>**Rating:**</span> {row['rating']} / 5.0", unsafe_allow_html=True)

                # Extract additional details
                if 'platforms' in game_details and game_details['platforms']:
                    platforms = [platform['platform']['name'] for platform in game_details['platforms']]
                    st.write(f"<span class='platform'>**Platforms:**</span> {', '.join(platforms)}", unsafe_allow_html=True)
                else:
                    st.write("<span class='platform'>**Platforms:**</span> Not available", unsafe_allow_html=True)

                if 'metacritic' in game_details:
                    st.write(f"<span class='metacritic'>**Metacritic Score:**</span> {game_details['metacritic']}", unsafe_allow_html=True)
                else:
                    st.write("<span class='metacritic'>**Metacritic Score:**</span> Not available", unsafe_allow_html=True)

                if 'genres' in game_details and game_details['genres']:
                    genres = [genre['name'] for genre in game_details['genres']]
                    st.write(f"<span class='genre'>**Genres:**</span> {', '.join(genres)}", unsafe_allow_html=True)
                else:
                    st.write("<span class='genre'>**Genres:**</span> Not available", unsafe_allow_html=True)

                if 'stores' in game_details and game_details['stores']:
                    stores = [store['store']['name'] for store in game_details['stores']]
                    st.write(f"<span class='platform'>**Available on:**</span> {', '.join(stores)}", unsafe_allow_html=True)
                else:
                    st.write("<span class='platform'>**Available on:**</span> Not available", unsafe_allow_html=True)

                if 'tags' in game_details and game_details['tags']:
                    tags = [tag['name'] for tag in game_details['tags']]
                    st.write(f"<span class='tag'>Tags:</span> {', '.join(tags)}", unsafe_allow_html=True)
                else:
                    st.write("<span class='tag'>Tags:</span> Not available", unsafe_allow_html=True)

                if 'esrb_rating' in game_details and game_details['esrb_rating']:
                    st.write(f"<span class='esrb'>**ESRB Rating:**</span> {game_details['esrb_rating']['name']}", unsafe_allow_html=True)
                else:
                    st.write("<span class='esrb'>**ESRB Rating:**</span> Not available", unsafe_allow_html=True)

                if 'short_screenshots' in game_details and game_details['short_screenshots']:
                    st.write("**Screenshots:**")
                    for screenshot in game_details['short_screenshots']:
                        st.image(screenshot['image'], width=200)

                # Fetch trailers
                trailers = fetch_game_trailers(row['id'])
                if trailers and trailers.get('results'):
                    trailer_url = trailers['results'][0]['data']['max']
                    if trailer_url:
                        # Add expander for trailer
                        with st.expander("📽️ Watch Trailer"):
                            st.video(trailer_url)
                    else:
                        st.write("**Trailer not available.**")
                else:
                    st.write("**No trailers found for this game.**")
                
            st.markdown("---")  # Separator between games


    else:
        st.error("Game not found. Please try a different title.")

# Statistical Overview Expander
with st.expander("📊 Statistical Overview"):
    st.write(f"🔢 **Total Games Available:** {len(games_df)}")
    st.write(f"🎮 **Unique Genres:** {games_df['genres'].nunique()}")
    st.write(f"⭐ **Average Rating:** {games_df['rating'].mean():.2f}")
    # Features count and names
    features = games_df.columns.tolist()
    st.write(f"📋 **Total Features Count:** {len(features)}")
    st.write("📝 **Feature Names:**")
    st.write(", ".join(features))



# Footer
st.write("### 🌟 Developed by Team C18")
st.write("🏫 Institution: SRM Institute of Science And Technology")
st.write("👨‍🏫 Guide: Dr. S. Prasanna Devi (HOD-CSE & Professor)")
st.write("👩‍🎓 Members:")
st.write("- S. Vijay Krishna Sundaran - RA2111026040042")
st.write("- Prawin K - RA2111026040041")
st.write("- Srilakshan A - RA2111026040032")
# --- Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import ast  
import random 
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import time 

# --- Surprise library for SVD ---
try:
    from surprise import Dataset, Reader, SVD
except ImportError:
    st.error("Error: The 'scikit-surprise' library is required but not found.")
    st.error("Please install it: pip install scikit-surprise")
    st.error("And add 'scikit-surprise' to your requirements.txt file.")
    st.stop()

# --- Configuration ---
DATA_PATH = 'Data/final_df.csv'
NUM_REC_OPTIONS = [10, 25, 50, 100]
DEFAULT_TOP_N = NUM_REC_OPTIONS[0]
MOCK_USERS = 150
MOCK_RATINGS_PER_USER = 40
BUDGET_FRIENDLY_THRESHOLD = 400
TOP_N_LOCATIONS_PLOT = 15

# Define essential columns expected/needed from the input CSV
EXPECTED_COLUMNS = [
    'name', 'location', 'cuisines',
    'rating', 'rating_count', 'cost_for_two'
]

# --- Data Loading and Processing ---
@st.cache_data # Cache the output: DataFrame and lists
def load_and_prepare_data(path):
    """
    Loads restaurant data, validates, cleans, parses cuisines,
    calculates popularity score, and extracts unique filter values.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Data file not found at '{path}'. Check path.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error: Could not load data file: {e}")
        st.stop()

    # --- Data Validation ---
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Fatal Error: CSV missing essential columns: {', '.join(missing_cols)}.")
        st.error("Ensure CSV has name, location, cuisines, rating, rating_count, cost_for_two.")
        st.stop()

    # --- Data Cleaning & Preprocessing ---
    try:
        # Fill missing text fields
        for col in ['cuisines', 'location']: # Removed place_types
             if col in df.columns: df[col] = df[col].fillna('')
             else: df[col] = '' # Should be caught by validation

        # Clean numeric columns
        for col, default_median in [('cost_for_two', 500), ('rating', 3.5), ('rating_count', 50)]:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
                  median_val = df[col].median()
                  df[col] = df[col].fillna(median_val if pd.notna(median_val) else default_median)
             else: df[col] = default_median # Should be caught by validation

        # Parse cuisines string into list
        def parse_cuisines(cuisine_str):
            if isinstance(cuisine_str, list): return cuisine_str
            if not isinstance(cuisine_str, str) or not cuisine_str.strip(): return []
            try:
                if cuisine_str.startswith('[') and cuisine_str.endswith(']'):
                    parsed_list = ast.literal_eval(cuisine_str)
                    return [str(item).strip() for item in parsed_list if str(item).strip()] if isinstance(parsed_list, list) else []
                else: return [c.strip() for c in cuisine_str.split(',') if c.strip()]
            except: return [cuisine_str.strip()] if cuisine_str.strip() else []
            return []
        df['cuisines_list'] = df['cuisines'].apply(parse_cuisines)

        # --- Calculate Popularity Score ---
        scaler = MinMaxScaler()
        try: df[['normalized_rating', 'normalized_rating_count']] = scaler.fit_transform(df[['rating', 'rating_count']])
        except ValueError: df['normalized_rating'], df['normalized_rating_count'] = 0.5, 0.5 # Fallback
        try:
            cost_range = df['cost_for_two'].max() - df['cost_for_two'].min()
            if cost_range > 0: df['normalized_cost'] = 1 - scaler.fit_transform(df[['cost_for_two']])
            else: df['normalized_cost'] = 0.5
        except ValueError: df['normalized_cost'] = 0.5 # Fallback
        df['popularity_score'] = (df['normalized_rating'] * 0.6 + df['normalized_rating_count'] * 0.3 + df['normalized_cost'] * 0.1)

        # Extract unique values for filters
        unique_locations = sorted(df['location'].astype(str).fillna('Unknown').unique())
        all_cuisines = sorted(list(set(cuisine for sublist in df['cuisines_list'] for cuisine in sublist)))

        # Ensure unique index
        if not df.index.is_unique: df = df.reset_index(drop=True)

        return df, unique_locations, all_cuisines

    except Exception as e:
        st.error(f"Error during data preparation: {e}")
        st.stop()

# --- Mock User Rating Generation ---
@st.cache_data
def generate_mock_ratings(_df, num_users, num_ratings_per_user):
    ratings_list = []
    all_restaurant_ids = _df.index.tolist()
    if not all_restaurant_ids: return pd.DataFrame(columns=['user_id', 'restaurant_id', 'rating'])
    for user_id_num in range(num_users):
        user_id = f'mock_user_{user_id_num}'
        num_to_rate = min(num_ratings_per_user, len(all_restaurant_ids))
        if num_to_rate <= 0: continue
        try: rated_restaurant_ids = random.sample(all_restaurant_ids, num_to_rate)
        except ValueError: continue
        user_bias = random.gauss(0, 0.3)
        for rest_id in rated_restaurant_ids:
            try:
                base_rating = _df.loc[rest_id, 'rating']
                if not isinstance(base_rating, (int, float)): base_rating = 3.0
                noise = random.gauss(0, 0.5)
                mock_rating = np.clip(base_rating + user_bias + noise, 1.0, 5.0)
                final_rating = round(mock_rating * 2) / 2
                ratings_list.append({'user_id': user_id, 'restaurant_id': rest_id, 'rating': final_rating})
            except KeyError: continue
    return pd.DataFrame(ratings_list)

# --- SVD Model Training ---
@st.cache_resource
def train_svd_model(ratings_df):
    if ratings_df.empty or not all(col in ratings_df.columns for col in ['user_id', 'restaurant_id', 'rating']): return None
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'restaurant_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
        svd_model.fit(trainset)
        return svd_model
    except Exception as e:
        st.error(f"Error during SVD model training: {e}")
        return None

# --- TF-IDF Calculation ---
@st.cache_resource
def fit_tfidf(_df):
    """ Fits TF-IDF on combined content features: cuisines, location. """
    try:
        # Combine features: cuisines and location
        feature_cols = ['cuisines', 'location'] # Removed place_types
        if not all(col in _df.columns for col in feature_cols):
             st.error(f"Missing columns for TF-IDF: {', '.join(feature_cols)}")
             st.stop()
        _df['content_features'] = _df[feature_cols].astype(str).agg(' '.join, axis=1)

        # Initialize and fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(_df['content_features'])
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"TF-IDF calculation error: {e}")
        st.stop()

# --- Content Similarity Matrix Calculation ---
@st.cache_resource # Cache the large similarity matrix
def calculate_content_similarity(_tfidf_matrix):
    """ Calculates the cosine similarity matrix from the TF-IDF matrix. """
    try:
        if _tfidf_matrix is None:
             return None
        start_time = time.time()
        cosine_sim = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
        end_time = time.time()
        return cosine_sim
    except Exception as e:
        st.error(f"Error calculating content similarity: {e}")
        return None

# --- Recommendation Logic Functions ---

def get_content_recommendations(df_filtered, selected_cuisines_list, vectorizer, tfidf_matrix_full, df_full_index, top_n):
    """ Recommends restaurants based on content similarity to selected cuisines. """
    if not selected_cuisines_list: return pd.DataFrame()
    if tfidf_matrix_full is None or vectorizer is None: return pd.DataFrame()
    try:
        user_profile_str = ' '.join(selected_cuisines_list)
        user_tfidf = vectorizer.transform([user_profile_str])
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix_full).flatten()
        all_scores = pd.Series(cosine_similarities, index=df_full_index)
        filtered_indices = df_filtered.index
        relevant_scores = all_scores[all_scores.index.isin(filtered_indices)]
        top_indices = relevant_scores.nlargest(top_n).index
        if top_indices.empty: return pd.DataFrame() # Handle case where no relevant scores found
        recommendations = df_filtered.loc[top_indices].copy()
        recommendations['score'] = relevant_scores.loc[top_indices]
        return recommendations.sort_values(by='score', ascending=False)
    except Exception as e: st.error(f"Content recommendation error: {e}"); return pd.DataFrame()

def get_recommendations_similar_to(restaurant_index, content_sim_matrix, df_full, top_n):
    """ Gets top N recommendations based on pre-calculated content similarity matrix. """
    if content_sim_matrix is None or restaurant_index is None or restaurant_index >= content_sim_matrix.shape[0]:
        return pd.DataFrame()
    try:
        sim_scores = list(enumerate(content_sim_matrix[restaurant_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        restaurant_indices = [i[0] for i in sim_scores]
        recommendations = df_full.iloc[restaurant_indices].copy()
        recommendations['similarity_score'] = [s[1] for s in sim_scores]
        return recommendations
    except Exception as e:
        st.error(f"Error finding similar restaurants: {e}")
        return pd.DataFrame()


def get_popularity_recommendations(df_filtered, top_n):
    """ Recommends restaurants based purely on the calculated 'popularity_score'. """
    try:
        if 'popularity_score' not in df_filtered.columns: return pd.DataFrame()
        recommendations = df_filtered.sort_values(by='popularity_score', ascending=False).head(top_n).copy()
        recommendations['score'] = recommendations['popularity_score']
        return recommendations
    except Exception as e: st.error(f"Popularity recommendation error: {e}"); return pd.DataFrame()


def get_svd_recommendations(target_user_id, df_filtered, svd_model, df_full, top_n):
    """ Generates recommendations using the trained SVD model for restaurants in df_filtered. """
    if svd_model is None: return pd.DataFrame()
    if df_filtered.empty: return pd.DataFrame()
    restaurant_ids_to_predict = df_filtered.index.tolist()
    predictions = []
    all_valid_ids = df_full.index
    for rest_id in restaurant_ids_to_predict:
        if rest_id in all_valid_ids:
            try:
                pred = svd_model.predict(uid=target_user_id, iid=rest_id)
                predictions.append({'restaurant_id': rest_id, 'score': pred.est})
            except: continue
    if not predictions: return pd.DataFrame()
    pred_df = pd.DataFrame(predictions).set_index('restaurant_id')
    recs = df_filtered.join(pred_df, how='inner')
    return recs.sort_values('score', ascending=False).head(top_n)


def combine_recommendations(df_filtered, recs_dict, weights, top_n):
    """ Combines recommendations using weighted rank aggregation. """
    try:
        final_scores = {}
        pool_sizes = [len(rec_df) for rec_df in recs_dict.values() if isinstance(rec_df, pd.DataFrame) and not rec_df.empty]
        max_rank_score = max(pool_sizes) if pool_sizes else top_n
        for method, rec_df in recs_dict.items():
            weight = weights.get(method, 0)
            if weight > 0 and isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
                for rank, index in enumerate(rec_df.index):
                    if index in df_filtered.index:
                        rank_score = max_rank_score - rank
                        current_score = final_scores.get(index, 0)
                        final_scores[index] = current_score + (weight * rank_score)
        if not final_scores: return pd.DataFrame()
        sorted_indices = sorted(final_scores, key=final_scores.get, reverse=True)
        top_indices = [idx for idx in sorted_indices if idx in df_filtered.index][:top_n]
        if not top_indices: return pd.DataFrame()
        recommendations = df_filtered.loc[top_indices].copy()
        recommendations['hybrid_rank_score'] = recommendations.index.map(final_scores)
        return recommendations.sort_values('hybrid_rank_score', ascending=False)
    except Exception as e: st.error(f"Hybrid combination error: {e}"); return pd.DataFrame()


# --- Helper Functions ---
@st.cache_data
def dataframe_to_csv(df):
    """ Converts a DataFrame to CSV bytes for downloading. """
    try: return df.to_csv(index=False).encode('utf-8')
    except Exception as e: st.error(f"CSV conversion error: {e}"); return None

# --- UI Rendering Functions ---

def render_recommendations_page(df, unique_locations, all_cuisines, vectorizer, tfidf_matrix_full, content_sim_matrix, svd_model, target_user_id):
    """ Renders the main recommendations page with filters and results. """
    st.header("Find Your Next Meal")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Options")
    location_options = ["All Locations"] + unique_locations
    selected_location = st.sidebar.selectbox("Location (Optional)", location_options, index=0)
    selected_cuisines = st.sidebar.multiselect("Cuisines (Optional)", all_cuisines)
    if not df.empty and pd.api.types.is_numeric_dtype(df['cost_for_two']):
        min_cost, max_cost_limit, median_cost = int(df['cost_for_two'].min()), int(df['cost_for_two'].max()), int(df['cost_for_two'].median())
        max_cost_filter = st.sidebar.slider("Max Cost for Two", min_value=min_cost, max_value=max_cost_limit, value=median_cost, disabled=(min_cost >= max_cost_limit))
    else: max_cost_filter = 500
    if not df.empty and pd.api.types.is_numeric_dtype(df['rating']):
        min_rating_limit, max_rating = float(df['rating'].min()), float(df['rating'].max())
        default_rating = 4.0 if 4.0 >= min_rating_limit and 4.0 <= max_rating else min_rating_limit
        min_rating_filter = st.sidebar.slider("Minimum Rating", min_value=min_rating_limit, max_value=max_rating, value=default_rating, step=0.1, disabled=(min_rating_limit >= max_rating))
    else: min_rating_filter = 3.0

    # --- Number of Recommendations Selector ---
    num_recommendations = st.sidebar.selectbox(
        "Number of results:",
        options=NUM_REC_OPTIONS,
        index=0, # Default to the first option (10)
        key="num_recs_select"
    )

    st.sidebar.markdown("---")

    # --- Recommendation Type and Weights ---
    st.sidebar.header("Recommendation Strategy")
    rec_type = st.sidebar.radio(
        "Choose Method:",
        ["Hybrid", "Content-Based", "Popularity", "Collaborative Filtering"],
        key="rec_type_radio"
    )

    weights = {}
    hybrid_help = """
    Adjust the influence of different factors:
    - **Content:** Similarity based on cuisines and location. (Requires selecting cuisines).
    - **Popularity:** Based on ratings, review counts, and cost.
    - **Collaborative Filtering:** Based on patterns learned from simulated user ratings (SVD model).
    """
    if rec_type == "Hybrid":
        st.sidebar.subheader("Hybrid Weights", help=hybrid_help)
        weights['content'] = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.4, 0.1, key="content_w")
        weights['popularity'] = st.sidebar.slider("Popularity Weight", 0.0, 1.0, 0.2, 0.1, key="pop_w")
        weights['cf'] = st.sidebar.slider("Collaborative Weight", 0.0, 1.0, 0.4, 0.1, key="cf_w") # Label simplified
        total_weight = sum(weights.values())
        if total_weight > 0: weights = {k: v / total_weight for k, v in weights.items()}
        else: weights = {'content': 0.4, 'popularity': 0.2, 'cf': 0.4}

    # --- Apply Filters ---
    try:
        filtered_df = df.copy()
        if selected_location != "All Locations": filtered_df = filtered_df[filtered_df['location'] == selected_location]
        if 'cost_for_two' in filtered_df.columns: filtered_df = filtered_df[filtered_df['cost_for_two'] <= max_cost_filter]
        if 'rating' in filtered_df.columns: filtered_df = filtered_df[filtered_df['rating'] >= min_rating_filter]
        if selected_cuisines:
            if 'cuisines_list' in filtered_df.columns:
                 mask = filtered_df['cuisines_list'].apply(lambda lst: isinstance(lst, list) and any(c in lst for c in selected_cuisines))
                 filtered_df = filtered_df[mask]
        filtered_df = filtered_df.copy()
    except Exception as filter_error:
        st.error(f"Error applying filters: {filter_error}")
        filtered_df = pd.DataFrame()

    # --- Generate Recommendations ---
    recommendations_df = pd.DataFrame()
    if not filtered_df.empty:
        # Use the user-selected number of recommendations
        n_recs = num_recommendations
        n_pool = num_recommendations * 2 # Pool size for hybrid ranking

        if rec_type == "Content-Based":
            if selected_cuisines: recommendations_df = get_content_recommendations(filtered_df, selected_cuisines, vectorizer, tfidf_matrix_full, df.index, n_recs)
            else: st.warning("Please select cuisines for Content-Based recommendations.")
        elif rec_type == "Popularity":
            recommendations_df = get_popularity_recommendations(filtered_df, n_recs)
        elif rec_type == "Collaborative Filtering": 
            if svd_model and target_user_id: recommendations_df = get_svd_recommendations(target_user_id, filtered_df, svd_model, df, n_recs)
            else: st.warning("SVD Model unavailable.")
        elif rec_type == "Hybrid":
            recs = {}
            if selected_cuisines: recs['content'] = get_content_recommendations(filtered_df, selected_cuisines, vectorizer, tfidf_matrix_full, df.index, n_pool)
            else: recs['content'] = pd.DataFrame(index=[])
            recs['popularity'] = get_popularity_recommendations(filtered_df, n_pool)
            if svd_model and target_user_id: recs['cf'] = get_svd_recommendations(target_user_id, filtered_df, svd_model, df, n_pool)
            else: recs['cf'] = pd.DataFrame(index=[])
            recommendations_df = combine_recommendations(filtered_df, recs, weights, n_recs)
    else:
         st.info("No restaurants match the current filters.")

    # --- Display Area ---
    st.markdown("---")
    st.subheader(f"Top {len(recommendations_df) if not recommendations_df.empty else 0} Recommendations")

    search_term = st.text_input("Search within recommendations by name:", key="recommendation_search")

    if not recommendations_df.empty:
        if search_term:
            try: recommendations_to_display = recommendations_df[recommendations_df['name'].str.contains(search_term, case=False, na=False)]
            except Exception: recommendations_to_display = recommendations_df
        else: recommendations_to_display = recommendations_df

        if not recommendations_to_display.empty:
            display_cols = ['name', 'cuisines', 'rating', 'rating_count', 'cost_for_two', 'location']
            # if 'place_types' in recommendations_to_display.columns: display_cols.append('place_types')
            # if 'address' in recommendations_to_display.columns: display_cols.append('address')
            display_cols = [col for col in display_cols if col in recommendations_to_display.columns] # Ensure they exist

            st.dataframe(recommendations_to_display[display_cols].reset_index(drop=True))

            csv_data = dataframe_to_csv(recommendations_to_display[display_cols])
            if csv_data:
                st.download_button(label="📥 Download Results as CSV", data=csv_data, file_name='recommendations.csv', mime='text/csv')

            st.markdown("---")
            st.subheader("Find Restaurants Similar to One You Like")
            # Create a list of restaurant names for the selectbox
            restaurant_names = [""] + sorted(df['name'].astype(str).unique()) # Add blank option
            selected_name = st.selectbox("Select a restaurant:", restaurant_names, index=0, key="similar_restaurant_select")

            if selected_name and content_sim_matrix is not None:
                try:
                    # Find the index of the selected restaurant
                    target_index = df[df['name'] == selected_name].index
                    if not target_index.empty:
                        target_index = target_index[0] # Get the first match index
                        # Get recommendations based on content similarity
                        similar_recs = get_recommendations_similar_to(target_index, content_sim_matrix, df, num_recommendations) # Use selected N

                        if not similar_recs.empty:
                            st.write(f"Restaurants similar to **{selected_name}**:")
                            # Display similar restaurants
                            sim_display_cols = ['name', 'cuisines', 'rating', 'cost_for_two', 'location']
                            sim_display_cols = [col for col in sim_display_cols if col in similar_recs.columns]
                            st.dataframe(similar_recs[sim_display_cols].reset_index(drop=True))
                        else:
                            st.info(f"Could not find similar restaurants for {selected_name}.")
                    else:
                        st.warning(f"Could not find restaurant named '{selected_name}' in the dataset.")
                except Exception as sim_error:
                    st.error(f"Error finding similar restaurants: {sim_error}")


            # --- Visualizations for Recommendations ---
            st.markdown("---")
            st.subheader("Insights from Recommendations")
            col1, col2 = st.columns(2)
            with col1:
                avg_rating = recommendations_to_display['rating'].mean() if 'rating' in recommendations_to_display and pd.api.types.is_numeric_dtype(recommendations_to_display['rating']) else None
                avg_cost = recommendations_to_display['cost_for_two'].mean() if 'cost_for_two' in recommendations_to_display and pd.api.types.is_numeric_dtype(recommendations_to_display['cost_for_two']) else None
                st.metric("Average Rating", f"{avg_rating:.2f} ⭐" if pd.notna(avg_rating) else "N/A")
                st.metric("Average Cost (for Two)", f"₹{avg_cost:.0f}" if pd.notna(avg_cost) else "N/A")
            with col2:
                try:
                    if 'cuisines_list' in recommendations_to_display.columns:
                        rec_cuisines = [c for sublist in recommendations_to_display['cuisines_list'] for c in sublist]
                        if rec_cuisines:
                            cuisine_counts = pd.Series(rec_cuisines).value_counts().nlargest(10)
                            cuisine_df = cuisine_counts.reset_index()
                            cuisine_df.columns = ['Cuisine', 'Count']
                            fig_rec_cuisine = px.bar(cuisine_counts, x=cuisine_counts.index, y=cuisine_counts.values, title="Top Cuisines in Recommendations", labels={'x': 'Cuisine', 'y': 'Count'})
                            fig_rec_cuisine.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_rec_cuisine, use_container_width=True)
                        else: st.caption("No cuisine data.")
                    else: st.caption("Cuisine list data missing.")
                except Exception as plot_error: st.warning(f"Could not plot cuisines: {plot_error}")
        else:
             st.info(f"No recommendations match '{search_term}'.")
   


def render_overview_page(df, all_cuisines):
    """ Renders the data overview page with KPIs and overall visualizations. """
    st.header("Dataset Overview")
    if df.empty: st.warning("No data available."); return

    # --- KPIs ---
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    try:
        total_restaurants, unique_locs, unique_cui = len(df), df['location'].nunique(), len(all_cuisines)
        avg_rating = df['rating'].mean() if 'rating' in df and pd.api.types.is_numeric_dtype(df['rating']) else None
        avg_cost = df['cost_for_two'].mean() if 'cost_for_two' in df and pd.api.types.is_numeric_dtype(df['cost_for_two']) else None
        budget_friendly_count = df[df['cost_for_two'] <= BUDGET_FRIENDLY_THRESHOLD].shape[0] if 'cost_for_two' in df and pd.api.types.is_numeric_dtype(df['cost_for_two']) else 0
        percent_budget = (budget_friendly_count / total_restaurants * 100) if total_restaurants > 0 else 0
        col1.metric("Total Restaurants", f"{total_restaurants:,}")
        col2.metric("Unique Locations", f"{unique_locs:,}")
        col3.metric("Unique Cuisines", f"{unique_cui:,}")
        col1.metric("Average Rating", f"{avg_rating:.2f} ⭐" if pd.notna(avg_rating) else "N/A")
        col2.metric("Average Cost (for Two)", f"₹{avg_cost:.0f}" if pd.notna(avg_cost) else "N/A")
        col3.metric(f"Budget Friendly (<₹{BUDGET_FRIENDLY_THRESHOLD})", f"{percent_budget:.1f}%")
    except Exception as kpi_error: st.warning(f"KPI calculation error: {kpi_error}")

    st.markdown("---")
    st.subheader("Overall Data Distributions")
    col_a, col_b = st.columns(2)
    try:
        with col_a: 
            # Rating Distribution
            if 'rating' in df and pd.api.types.is_numeric_dtype(df['rating']):
                fig_rating = px.histogram(df, x='rating', nbins=20, title="Overall Rating Distribution")
                st.plotly_chart(fig_rating, use_container_width=True)
            # Top Locations Bar Chart
            if 'location' in df:
                 top_locations = df['location'].value_counts().nlargest(TOP_N_LOCATIONS_PLOT).sort_values(ascending=True)
                 if not top_locations.empty:
                      fig_loc = px.bar(top_locations, y=top_locations.index, x=top_locations.values, orientation='h', title=f"Top {TOP_N_LOCATIONS_PLOT} Locations by Restaurant Count", labels={'y': 'Location', 'x': '# Restaurants'})
                      fig_loc.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                      st.plotly_chart(fig_loc, use_container_width=True)
        with col_b: 
            # Cost Distribution
            if 'cost_for_two' in df and pd.api.types.is_numeric_dtype(df['cost_for_two']):
                fig_cost = px.histogram(df, x='cost_for_two', nbins=20, title="Overall Cost for Two Distribution (INR)")
                st.plotly_chart(fig_cost, use_container_width=True)
            # --- Rating Distribution by Location (Box Plot) ---
            if 'rating' in df and 'location' in df:
                st.markdown("---") # Separator
                st.subheader(f"Rating Distribution by Top {TOP_N_LOCATIONS_PLOT} Locations")
                # Get top N locations by count
                top_loc_names = df['location'].value_counts().nlargest(TOP_N_LOCATIONS_PLOT).index.tolist()
                # Filter dataframe for only these locations
                df_top_locs = df[df['location'].isin(top_loc_names)]
                if not df_top_locs.empty:
                    fig_box = px.box(df_top_locs, x='location', y='rating', title=f"Rating Distribution for Top {TOP_N_LOCATIONS_PLOT} Locations",
                                     labels={'location': 'Location', 'rating': 'Rating'},
                                     points=False) # 'outliers', False, or 'all'
                    fig_box.update_layout(xaxis={'categoryorder':'total descending'}) # Order by count implicitly
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                     st.caption("Not enough data for location rating distribution.")

    except Exception as plot_error: st.warning(f"Overview plot error: {plot_error}")

    # --- Cuisine Word Cloud ---
    st.markdown("---")
    st.subheader("Cuisine Word Cloud")
    try:
        all_cuisine_text = ' '.join(str(cuisine) for sublist in df['cuisines_list'] for cuisine in sublist if cuisine)
        if all_cuisine_text:
            try:
                wordcloud = WordCloud(width=800, height=350, background_color='white', collocations=False).generate(all_cuisine_text)
                fig, ax = plt.subplots(figsize=(12, 6)) # Adjusted figure size
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as wc_gen_err: st.warning(f"Word cloud generation error: {wc_gen_err}")
        else: st.caption("Not enough cuisine data for word cloud.")
    except ImportError: st.warning("Install 'wordcloud' and 'matplotlib'.")
    except Exception as wc_error: st.warning(f"Word cloud error: {wc_error}")


# --- Main App Execution Logic ---

# Initialize session state if not already done
if 'app_initialized' not in st.session_state:
    st.session_state.update({
        'app_initialized': False, 'df_restaurants': pd.DataFrame(), 'unique_locations': [],
        'all_cuisines': [], 'vectorizer': None, 'tfidf_matrix': None,
        'content_sim_matrix': None, 'svd_model': None, 'target_svd_user': None
    })

# Load data and train models only once
if not st.session_state.app_initialized:
    with st.spinner("Initializing application... Please wait."):
        try:
            df_restaurants, unique_locations, all_cuisines = load_and_prepare_data(DATA_PATH)
            vectorizer, tfidf_matrix = fit_tfidf(df_restaurants)
            # Calculate content similarity matrix after fitting TF-IDF
            content_sim_matrix = calculate_content_similarity(tfidf_matrix)
            # Generate mock data and train SVD
            mock_ratings_data = generate_mock_ratings(df_restaurants, MOCK_USERS, MOCK_RATINGS_PER_USER)
            svd_model_trained = train_svd_model(mock_ratings_data)
            target_svd_user = 'mock_user_0' if not mock_ratings_data.empty else None

            # Store loaded data and models in session state
            st.session_state.update({
                'df_restaurants': df_restaurants, 'unique_locations': unique_locations,
                'all_cuisines': all_cuisines, 'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix, 'content_sim_matrix': content_sim_matrix,
                'svd_model': svd_model_trained, 'target_svd_user': target_svd_user,
                'app_initialized': True
            })
            if hasattr(st, 'toast'): st.toast("Application ready!", icon="🎉")
        except SystemExit: pass # Allow st.stop() to work
        except Exception as startup_error:
             st.error(f"Critical startup error: {startup_error}")
             st.exception(startup_error)
             st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Recommendations", "Data Overview"], key="navigation_radio")
st.sidebar.markdown("---")

# --- Page Rendering ---
if st.session_state.app_initialized:
    try:
        if page == "Recommendations":
            render_recommendations_page(
                df=st.session_state.df_restaurants,
                unique_locations=st.session_state.unique_locations,
                all_cuisines=st.session_state.all_cuisines,
                vectorizer=st.session_state.vectorizer,
                tfidf_matrix_full=st.session_state.tfidf_matrix, # Pass the full matrix
                content_sim_matrix=st.session_state.content_sim_matrix, # Pass similarity matrix
                svd_model=st.session_state.svd_model,
                target_user_id=st.session_state.target_svd_user
            )
        elif page == "Data Overview":
            render_overview_page(
                df=st.session_state.df_restaurants,
                all_cuisines=st.session_state.all_cuisines
            )
    except Exception as page_render_error:
         st.error("An error occurred while rendering the page.")
         st.exception(page_render_error)
else:
     st.error("Application failed to initialize. Please check data file and dependencies.")

# --- About Section ---
st.sidebar.markdown("---")
with st.sidebar.expander("About this App"):
    st.markdown("""
        **Restaurant Recommendation Engine**

        Discover restaurants using various recommendation strategies.

        **Recommendation Strategies:**
        * **Hybrid:** Blends Content, Popularity, and Collaborative signals. Use the '?' icon above weights for details.
        * **Content-Based:** Uses TF-IDF on cuisines and location. Requires selecting cuisines.
        * **Popularity:** Ranks by a score based on ratings, reviews, and cost.
        * **Collaborative Filtering:** Uses an SVD model trained on simulated user ratings.

        **Features:**
        * Filter by location, cuisine, cost, rating.
        * Select number of results.
        * Search within recommendations.
        * Find restaurants similar to one you like.
        * Export recommendations to CSV.
        * View overall data insights on the 'Data Overview' page.

        *Note: Uses simulated data for CF. Production systems need real data & validation.*
    """)

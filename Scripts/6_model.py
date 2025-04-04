import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import random
import pickle

df = pd.read_csv('../Data/final_df.csv')

# Content-based filtering component
df['content_features'] = df['cuisines'].fillna('') + ' ' + df['location'].fillna('') + ' ' + df['place_types'].fillna('')
df['content_features'] = df['content_features'].astype(str)

print("\nGenerating TF-IDF vectors...")
tfidf = TfidfVectorizer(min_df=2, max_features=5000, 
                      strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 2),
                      stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['content_features'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

print("Calculating content-based similarity...")
content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix shape: {content_sim_matrix.shape}")

# Create popularity score
scaler = MinMaxScaler()
df[['normalized_rating', 'normalized_rating_count']] = scaler.fit_transform(df[['rating', 'rating_count']])
# Lesser cost is better, so we invert the cost
df['normalized_cost'] = 1 - scaler.fit_transform(df[['cost_for_two']])
df['popularity_score'] = (df['normalized_rating'] * 0.6 + 
                         df['normalized_rating_count'] * 0.3 + 
                         df['normalized_cost'] * 0.1)

df.to_csv('../Data/data_for_feature_importance.csv', index=False)

# Feature-based filtering functions
def get_restaurants_by_location(location, top_n=10):
    filtered_df = df[df['location'] == location]
    return filtered_df.sort_values('popularity_score', ascending=False).head(top_n)

def get_restaurants_by_budget(max_budget, top_n=10):
    filtered_df = df[df['cost_for_two'] <= max_budget]
    return filtered_df.sort_values('popularity_score', ascending=False).head(top_n)

def get_restaurants_by_cuisine(cuisine, top_n=10):
    filtered_df = df[df['cuisines'].str.contains(cuisine, case=False, na=False)]
    return filtered_df.sort_values('popularity_score', ascending=False).head(top_n)

def get_restaurants_by_rating_minimum(min_rating, top_n=10):
    filtered_df = df[df['rating'] >= min_rating]
    return filtered_df.sort_values('popularity_score', ascending=False).head(top_n)

# Content-based recommendation function
# Takes a restaurant ID, iterates over the similarity matrix, and returns the top N similar restaurants
# Excludes the restaurant itself from the recommendations

def get_content_based_recommendations(restaurant_id, sim_matrix, top_n=10):
    sim_scores = list(enumerate(sim_matrix[restaurant_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    restaurant_indices = [i[0] for i in sim_scores]
    return df.iloc[restaurant_indices]

# Personalized recommendation function
# budget means max budget, rating_threshold means min rating

def get_personalized_recommendations(cuisine_prefs=None, location_pref=None, 
                                   budget=None, rating_threshold=None, 
                                   liked_restaurant_ids=None, sim_matrix=content_sim_matrix, 
                                   top_n=10):
    candidates = df.copy()
    
    if cuisine_prefs:
        cuisine_filter = '|'.join([f"(?i){cuisine}" for cuisine in cuisine_prefs])
        candidates = candidates[candidates['cuisines'].str.contains(cuisine_filter, regex=True, na=False)]
    
    if location_pref:
        candidates = candidates[candidates['location'] == location_pref]
    
    if budget:
        candidates = candidates[candidates['cost_for_two'] <= budget]
    
    if rating_threshold:
        candidates = candidates[candidates['rating'] >= rating_threshold]
    
    if liked_restaurant_ids and len(liked_restaurant_ids) > 0:
        all_similar_restaurants = pd.DataFrame()
        
        for rest_id in liked_restaurant_ids:
            similar_restaurants = get_content_based_recommendations(rest_id, sim_matrix, top_n=20)
            similar_restaurants['similarity_score'] = similar_restaurants.index.map(lambda idx: sim_matrix[rest_id][idx])
            all_similar_restaurants = pd.concat([all_similar_restaurants, similar_restaurants])
        
        avg_scores = all_similar_restaurants.groupby(all_similar_restaurants.index)['similarity_score'].mean()
        filtered_similar = candidates[candidates.index.isin(avg_scores.index)]
        
        filtered_similar['recommendation_score'] = filtered_similar.index.map(
            lambda idx: avg_scores.get(idx, 0) * 0.7 + filtered_similar.loc[idx, 'popularity_score'] * 0.3
        )
        
        return filtered_similar.sort_values('recommendation_score', ascending=False).head(top_n)
    
    return candidates.sort_values('popularity_score', ascending=False).head(top_n)

# Create collaborative filtering component
# Constructs a matrix where rows represent unique locations and columns represent restaurant indices.
# Each cell contains the rating given by that location to that restaurant.

def create_utility_matrix():
    locations = df['location'].unique()
    utility_matrix = pd.DataFrame(index=locations, columns=range(len(df)))
    
    for location in locations:
        location_restaurants = df[df['location'] == location]
        for idx, row in location_restaurants.iterrows():
            utility_matrix.loc[location, idx] = row['rating']
    
    utility_matrix = utility_matrix.fillna(0)
    return utility_matrix

utility_matrix = create_utility_matrix()
utility_np = utility_matrix.to_numpy()
U, sigma, Vt = svds(utility_np, k=min(50, min(utility_np.shape) - 1))
sigma_diag_matrix = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=utility_matrix.index, columns=utility_matrix.columns)

# Collaborative filtering recommendation function
def get_cf_recommendations(location, top_n=10):
    if location not in predicted_ratings_df.index:
        return df.sort_values('popularity_score', ascending=False).head(top_n)
    
    restaurant_indices = predicted_ratings_df.loc[location].sort_values(ascending=False).index
    location_restaurants = set(df[df['location'] == location].index)
    new_restaurants = [idx for idx in restaurant_indices if idx not in location_restaurants]
    
    return df.loc[new_restaurants[:top_n]]

# Hybrid recommendation function
# Composed of 3 things: overall popularity score, content-based similarity, and collaborative filtering

def get_hybrid_recommendations(cuisines=None, location=None, budget=None, 
                             liked_restaurant_ids=None, rating_threshold=None,
                             content_weight=0.6, pop_weight=0.3, cf_weight=0.1,
                             top_n=10):
    candidates = df.copy()
    
    if cuisines:
        cuisine_filter = '|'.join([f"(?i){cuisine}" for cuisine in cuisines])
        candidates = candidates[candidates['cuisines'].str.contains(cuisine_filter, regex=True, na=False)]
    
    if location:
        candidates = candidates[candidates['location'] == location]
    
    if budget:
        candidates = candidates[candidates['cost_for_two'] <= budget]
    
    if rating_threshold:
        candidates = candidates[candidates['rating'] >= rating_threshold]
    
    candidates['hybrid_score'] = candidates['popularity_score'] * pop_weight
    
    if location and cf_weight > 0:
        if location in predicted_ratings_df.index:
            for idx in candidates.index:
                if idx in predicted_ratings_df.columns:
                    pred_rating = predicted_ratings_df.loc[location, idx]
                    normalized_pred = (pred_rating - predicted_ratings_df.loc[location].min()) / \
                                  (predicted_ratings_df.loc[location].max() - predicted_ratings_df.loc[location].min() + 1e-10)
                    candidates.loc[idx, 'hybrid_score'] += normalized_pred * cf_weight
    
    if liked_restaurant_ids and len(liked_restaurant_ids) > 0 and content_weight > 0:
        for idx in candidates.index:
            avg_sim = np.mean([content_sim_matrix[rest_id][idx] for rest_id in liked_restaurant_ids])
            candidates.loc[idx, 'hybrid_score'] += avg_sim * content_weight
    elif content_weight > 0:
        top_restaurants = df.sort_values('popularity_score', ascending=False).head(5).index
        for idx in candidates.index:
            avg_sim = np.mean([content_sim_matrix[rest_id][idx] for rest_id in top_restaurants])
            candidates.loc[idx, 'hybrid_score'] += avg_sim * content_weight
    
    return candidates.sort_values('hybrid_score', ascending=False).head(top_n)

# Evaluation methodology
def evaluate_recommendations():
    test_scenarios = []
    test_restaurants = random.sample(range(len(df)), min(100, len(df)))
    
    for idx in test_restaurants:
        restaurant = df.iloc[idx]
        location = restaurant['location']
        cuisines = restaurant['cuisines'].split(',') if pd.notna(restaurant['cuisines']) else []
        cuisine = cuisines[0] if len(cuisines) > 0 else None
        
        content_recs = get_content_based_recommendations(idx, content_sim_matrix, top_n=10)
        
        if cuisine:
            personalized_recs = get_personalized_recommendations(
                cuisine_prefs=[cuisine], location_pref=None, budget=None, 
                liked_restaurant_ids=[idx], top_n=10
            )
        else:
            personalized_recs = get_personalized_recommendations(
                cuisine_prefs=None, location_pref=None, budget=None, 
                liked_restaurant_ids=[idx], top_n=10
            )
            
        hybrid_recs = get_hybrid_recommendations(
            cuisines=[cuisine] if cuisine else None, location=None,
            budget=None, liked_restaurant_ids=[idx], top_n=10
        )
        
        cf_recs = get_cf_recommendations(location, top_n=10)
        
        test_scenarios.append({
            'restaurant_id': idx,
            'actual_cuisines': cuisines,
            'actual_location': location,
            'actual_rating': restaurant['rating'],
            'content_recs': content_recs.index.tolist(),
            'personalized_recs': personalized_recs.index.tolist(),
            'hybrid_recs': hybrid_recs.index.tolist(),
            'cf_recs': cf_recs.index.tolist()
        })
    
    # Calculate evaluation metrics
    content_precision = []
    personalized_precision = []
    hybrid_precision = []
    cf_precision = []
    
    content_cuisine_match = []
    personalized_cuisine_match = []
    hybrid_cuisine_match = []
    cf_cuisine_match = []
    
    def calc_cuisine_match(rec_ids, actual_cuisines):
        if not rec_ids or not actual_cuisines:
            return 0
        matches = 0
        for rec_id in rec_ids:
            rec_cuisines = df.loc[rec_id, 'cuisines'].split(',') if pd.notna(df.loc[rec_id, 'cuisines']) else []
            if any(c in actual_cuisines for c in rec_cuisines):
                matches += 1
        return matches / len(rec_ids) if len(rec_ids) > 0 else 0
    
    for scenario in test_scenarios:
        actual_cuisines = scenario['actual_cuisines']
        actual_rating = scenario['actual_rating']
        
        if not actual_cuisines:
            continue
            
        # Calculate cuisine match rates
        if scenario['content_recs']:
            content_cuisine_match.append(calc_cuisine_match(scenario['content_recs'], actual_cuisines))
            matching_ratings = sum(1 for rec_id in scenario['content_recs'] 
                                  if df.loc[rec_id, 'rating'] >= actual_rating - 0.5)
            content_precision.append(matching_ratings / len(scenario['content_recs']))
            
        if scenario['personalized_recs']:
            personalized_cuisine_match.append(calc_cuisine_match(scenario['personalized_recs'], actual_cuisines))
            matching_ratings = sum(1 for rec_id in scenario['personalized_recs'] 
                                  if df.loc[rec_id, 'rating'] >= actual_rating - 0.5)
            personalized_precision.append(matching_ratings / len(scenario['personalized_recs']))
            
        if scenario['hybrid_recs']:
            hybrid_cuisine_match.append(calc_cuisine_match(scenario['hybrid_recs'], actual_cuisines))
            matching_ratings = sum(1 for rec_id in scenario['hybrid_recs'] 
                                  if df.loc[rec_id, 'rating'] >= actual_rating - 0.5)
            hybrid_precision.append(matching_ratings / len(scenario['hybrid_recs']))
            
        if scenario['cf_recs']:
            cf_cuisine_match.append(calc_cuisine_match(scenario['cf_recs'], actual_cuisines))
            matching_ratings = sum(1 for rec_id in scenario['cf_recs'] 
                                  if df.loc[rec_id, 'rating'] >= actual_rating - 0.5)
            cf_precision.append(matching_ratings / len(scenario['cf_recs']))
    
    # Average metrics
    avg_content_precision = np.mean(content_precision) if content_precision else 0
    avg_personalized_precision = np.mean(personalized_precision) if personalized_precision else 0
    avg_hybrid_precision = np.mean(hybrid_precision) if hybrid_precision else 0
    avg_cf_precision = np.mean(cf_precision) if cf_precision else 0
    
    avg_content_cuisine = np.mean(content_cuisine_match) if content_cuisine_match else 0
    avg_personalized_cuisine = np.mean(personalized_cuisine_match) if personalized_cuisine_match else 0
    avg_hybrid_cuisine = np.mean(hybrid_cuisine_match) if hybrid_cuisine_match else 0
    avg_cf_cuisine = np.mean(cf_cuisine_match) if cf_cuisine_match else 0
    
    print("\nEvaluation Results (Rating Precision):")
    print(f"Content-based: {avg_content_precision:.4f}")
    print(f"Personalized: {avg_personalized_precision:.4f}")
    print(f"Hybrid: {avg_hybrid_precision:.4f}")
    print(f"Collaborative filtering: {avg_cf_precision:.4f}")
    
    print("\nEvaluation Results (Cuisine Match Rate):")
    print(f"Content-based: {avg_content_cuisine:.4f}")
    print(f"Personalized: {avg_personalized_cuisine:.4f}")
    print(f"Hybrid: {avg_hybrid_cuisine:.4f}")
    print(f"Collaborative filtering: {avg_cf_cuisine:.4f}")
    
    # Visualize results
    methods = ['Content-based', 'Personalized', 'Hybrid', 'Collaborative']
    precision_scores = [avg_content_precision, avg_personalized_precision, avg_hybrid_precision, avg_cf_precision]
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=methods, y=precision_scores)
    plt.title('Recommendation Method Rating Precision Comparison')
    plt.ylabel('Precision Score')
    plt.xlabel('Recommendation Method')
    plt.ylim(0, 1)
    
    for i, v in enumerate(precision_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
    plt.savefig('recommendation_precision.png')
    plt.close()
    
    cuisine_scores = [avg_content_cuisine, avg_personalized_cuisine, avg_hybrid_cuisine, avg_cf_cuisine]
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=methods, y=cuisine_scores)
    plt.title('Recommendation Method Cuisine Match Rate Comparison')
    plt.ylabel('Cuisine Match Rate')
    plt.xlabel('Recommendation Method')
    plt.ylim(0, 1)
    
    for i, v in enumerate(cuisine_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
    plt.savefig('cuisine_match_rate.png')
    plt.close()
    
    return {
        'content_precision': avg_content_precision,
        'personalized_precision': avg_personalized_precision,
        'hybrid_precision': avg_hybrid_precision,
        'cf_precision': avg_cf_precision,
        'content_cuisine': avg_content_cuisine,
        'personalized_cuisine': avg_personalized_cuisine,
        'hybrid_cuisine': avg_hybrid_cuisine,
        'cf_cuisine': avg_cf_cuisine,
        'test_scenarios': test_scenarios
    }

# Evaluate the recommendation system
evaluation_results = evaluate_recommendations()

# Display function for restaurants
def display_restaurants(restaurants_df):
    for i, (idx, row) in enumerate(restaurants_df.iterrows(), 1):
        print(f"{i}. {row['name']} ({row['location']}) - {row['cuisines']} - ₹{row['cost_for_two']} - Rating: {row['rating']}/5 ({row['rating_count']} reviews)")

# Generate example recommendations
def get_example_recommendations():
    print("\n===== RESTAURANT RECOMMENDATION EXAMPLES =====\n")
    
    # Popular recommendations
    print("Example 1: Top 5 most popular restaurants overall")
    top_popular = df.sort_values('popularity_score', ascending=False).head(5)
    display_restaurants(top_popular)
    
    # Location-based recommendations
    sample_location = df['location'].value_counts().index[0]
    print(f"\nExample 2: Top 5 restaurants in {sample_location}")
    location_recs = get_restaurants_by_location(sample_location, top_n=5)
    display_restaurants(location_recs)
    
    # Cuisine-based recommendations
    sample_cuisine = "North Indian"
    print(f"\nExample 3: Top 5 {sample_cuisine} restaurants")
    cuisine_recs = get_restaurants_by_cuisine(sample_cuisine, top_n=5)
    display_restaurants(cuisine_recs)
    
    # Budget-based recommendations
    budget = 300
    print(f"\nExample 4: Top 5 restaurants with budget under ₹{budget} for two")
    budget_recs = get_restaurants_by_budget(budget, top_n=5)
    display_restaurants(budget_recs)
    
    # Content-based recommendations
    sample_restaurant_idx = df[df['rating'] >= 4.0].sample(1).index[0]
    sample_restaurant = df.loc[sample_restaurant_idx]
    print(f"\nExample 5: Top 5 restaurants similar to '{sample_restaurant['name']}' ({sample_restaurant['cuisines']})")
    content_recs = get_content_based_recommendations(sample_restaurant_idx, content_sim_matrix, top_n=5)
    display_restaurants(content_recs)
    
    # Personalized recommendations
    cuisines = ["South Indian"]
    budget = 400
    print(f"\nExample 6: Top 5 personalized recommendations for {', '.join(cuisines)} cuisine with budget ₹{budget}")
    personalized_recs = get_personalized_recommendations(cuisine_prefs=cuisines, budget=budget, top_n=5)
    display_restaurants(personalized_recs)
    
    # Hybrid recommendations
    liked_ids = [df[df['rating'] >= 4.5].sample(2).index.tolist()[0]]
    liked_restaurant = df.loc[liked_ids[0]]
    cuisines = [liked_restaurant['cuisines'].split(',')[0]] if pd.notna(liked_restaurant['cuisines']) else []
    print(f"\nExample 7: Top 5 hybrid recommendations for someone who likes '{liked_restaurant['name']}'")
    hybrid_recs = get_hybrid_recommendations(cuisines=cuisines, liked_restaurant_ids=liked_ids, top_n=5)
    display_restaurants(hybrid_recs)

# Run examples
get_example_recommendations()

# Evaluation summary
print("\n===== RECOMMENDATION SYSTEM EVALUATION SUMMARY =====\n")
print("Rating Precision (higher is better):")
print(f"Content-based: {evaluation_results['content_precision']:.4f}")
print(f"Personalized: {evaluation_results['personalized_precision']:.4f}")
print(f"Hybrid: {evaluation_results['hybrid_precision']:.4f}")
print(f"Collaborative filtering: {evaluation_results['cf_precision']:.4f}")

print("\nCuisine Match Rate (higher is better):")
print(f"Content-based: {evaluation_results['content_cuisine']:.4f}")
print(f"Personalized: {evaluation_results['personalized_cuisine']:.4f}")
print(f"Hybrid: {evaluation_results['hybrid_cuisine']:.4f}")
print(f"Collaborative filtering: {evaluation_results['cf_cuisine']:.4f}")

# Determine best approach
best_precision = max(
    evaluation_results['content_precision'],
    evaluation_results['personalized_precision'],
    evaluation_results['hybrid_precision'],
    evaluation_results['cf_precision']
)

best_cuisine = max(
    evaluation_results['content_cuisine'],
    evaluation_results['personalized_cuisine'],
    evaluation_results['hybrid_cuisine'],
    evaluation_results['cf_cuisine']
)

print("\nBest performing approaches:")
methods = {
    'content_precision': 'Content-based filtering',
    'personalized_precision': 'Personalized recommendations',
    'hybrid_precision': 'Hybrid recommendations',
    'cf_precision': 'Collaborative filtering'
}

for method, value in evaluation_results.items():
    if 'precision' in method and value == best_precision:
        print(f"- {methods[method]} has the best rating precision")
    elif 'cuisine' in method and value == best_cuisine:
        method_name = method.replace('_cuisine', '')
        print(f"- {methods[method_name + '_precision']} has the best cuisine match rate")

# Optimize weights
content_weight, cf_weight, pop_weight = 0.6, 0.1, 0.3

if evaluation_results['hybrid_precision'] >= evaluation_results['content_precision'] and \
   evaluation_results['hybrid_precision'] >= evaluation_results['cf_precision']:
    
    total_score = evaluation_results['content_precision'] + evaluation_results['cf_precision'] + \
                 evaluation_results['personalized_precision']
    
    if total_score > 0:
        content_weight = evaluation_results['content_precision'] / total_score
        cf_weight = evaluation_results['cf_precision'] / total_score
        pop_weight = evaluation_results['personalized_precision'] / total_score
        
        sum_weights = content_weight + cf_weight + pop_weight
        content_weight /= sum_weights
        cf_weight /= sum_weights
        pop_weight /= sum_weights

# Save model and data
recommender_data = {
    'content_sim_matrix': content_sim_matrix,
    'tfidf_vectorizer': tfidf,
    'predicted_ratings_df': predicted_ratings_df,
    'popularity_scores': df['popularity_score'],
    'evaluation_results': evaluation_results,
    'optimal_weights': {
        'content_weight': content_weight,
        'cf_weight': cf_weight,
        'pop_weight': pop_weight
    }
}

with open('restaurant_recommender.pkl', 'wb') as f:
    pickle.dump(recommender_data, f)

print("\nRecommender system model and data saved to 'restaurant_recommender.pkl'")

# Function to load model and make recommendations
def load_recommender_and_recommend(user_preferences, df):
    try:
        with open('restaurant_recommender.pkl', 'rb') as f:
            recommender_data = pickle.load(f)
        
        content_sim_matrix = recommender_data['content_sim_matrix']
        optimal_weights = recommender_data['optimal_weights']
        
        recommendations = get_hybrid_recommendations(
            cuisines=user_preferences.get('cuisines'),
            location=user_preferences.get('location'),
            budget=user_preferences.get('budget'),
            liked_restaurant_ids=user_preferences.get('liked_restaurant_ids'),
            rating_threshold=user_preferences.get('rating_threshold'),
            content_weight=optimal_weights['content_weight'],
            pop_weight=optimal_weights['pop_weight'],
            cf_weight=optimal_weights['cf_weight'],
            top_n=user_preferences.get('top_n', 10)
        )
        
        return recommendations
    
    except Exception as e:
        print(f"Error loading recommender: {e}")
        return df.sort_values('popularity_score', ascending=False).head(
            user_preferences.get('top_n', 10)
        )

# Summary statistics
print(f"\nTotal restaurants: {len(df)}")
print(f"Unique locations: {df['location'].nunique()}")
print(f"Unique cuisines: {df['primary_cuisine'].nunique() if 'primary_cuisine' in df.columns else 'N/A'}")
print("\nOptimal recommendation weights:")
print(f"Content-based: {content_weight:.2f}")
print(f"Collaborative filtering: {cf_weight:.2f}")
print(f"Popularity-based: {pop_weight:.2f}")
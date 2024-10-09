import pandas as pd

# Load the CSV files
reviews_df = pd.read_csv('../datasets/rotten_tomatoes_movie_reviews.csv')  # movie ids, reviews, score (positive/negative)
movies_df = pd.read_csv('../datasets/rotten_tomatoes_movies.csv')   # movie ids, movie names
imdb_df = pd.read_csv('../datasets/imdb-movies-dataset.csv')     # movie names, imdb scores

# Check for duplicates in the initial datasets
reviews_df = reviews_df.drop_duplicates(subset=['id', 'reviewText'])
movies_df = movies_df.drop_duplicates(subset=['id', 'title'])
imdb_df = imdb_df.drop_duplicates(subset=['Title'])

# Merge the data based on movie id
# Step 1: Merge reviews with movie names using movie id
combined_df = pd.merge(reviews_df, movies_df, on='id', how='inner')

# Step 2: Merge the result with IMDb scores using movie name
final_df = pd.merge(combined_df, imdb_df, left_on='title', right_on='Title', how='inner')

# Select only the columns we need: movie name, id, written review, positive/negative, and imdb score
final_df = final_df[['title', 'id', 'reviewText', 'scoreSentiment', 'Rating']]

# Remove duplicate rows after merging (if they exist)
final_df = final_df.drop_duplicates()

# Filter out rows with missing values in key columns
final_df = final_df.dropna(subset=['id', 'title', 'reviewText', 'scoreSentiment', 'Rating'])

# Save the combined result to a new CSV file
final_df.to_csv('combined_output.csv')

print("Combined CSV file has been created successfully.")

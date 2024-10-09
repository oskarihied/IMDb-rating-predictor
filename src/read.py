#forest model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
'''
reviews_df = pd.read_csv('./datasets/rotten_tomatoes_movie_reviews.csv')
movies_df = pd.read_csv('./datasets/rotten_tomatoes_movies.csv')
imdb_df = pd.read_csv('./datasets/imdb-movies-dataset.csv')


reviews_df = reviews_df.drop_duplicates(subset=['id', 'reviewText'])
movies_df = movies_df.drop_duplicates(subset=['id', 'title'])
imdb_df = imdb_df.drop_duplicates(subset=['Title'])


combined_df = pd.merge(reviews_df, movies_df, on='id', how='inner')
final_df = pd.merge(combined_df, imdb_df, left_on='title', right_on='Title',how='inner')

final_df = final_df[['title', 'id', 'reviewText', 'scoreSentiment', 'Rating']]
final_df = final_df.drop_duplicates()
final_df = final_df.dropna(subset=['id', 'title', 'reviewText','scoreSentiment', 'Rating'])

final_df.to_csv('combined.csv')
print("Combined CSV file has been created successfully.")'''
final_df = pd.read_csv('combined.csv')
# Jako ominaisuuksiin ja kohteeseen
final_df['Sentiment_Label'] = final_df['scoreSentiment'].apply(lambda x: 1 if x=="POSITIVE" else 0)

# Features and target selection for sentiment classification
from sklearn.utils import resample

# Example of upsampling the minority class
majority_class = final_df[final_df['Sentiment_Label'] == 1]
minority_class = final_df[final_df['Sentiment_Label'] == 0]

minority_upsampled = resample(minority_class,
                              replace=True,     # sample with replacement
                              n_samples=len(majority_class),  # match majority class
                              random_state=42)  # reproducible results

# Combine majority and upsampled minority
upsampled = pd.concat([majority_class, minority_upsampled])

# Prepare your features and labels from upsampled data
X_sentiment = upsampled['reviewText']
y_sentiment = upsampled['Sentiment_Label']

# Step 1: Convert raw text to a count matrix using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', max_features=30000)
X_counts_sentiment = count_vectorizer.fit_transform(X_sentiment)

# Step 2: Transform the count matrix to a TF-IDF representation using TfidfTransformer
tfidf_transformer_sentiment = TfidfTransformer()
X_tfidf_sentiment = tfidf_transformer_sentiment.fit_transform(X_counts_sentiment)

# Train-test split for sentiment classification
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
    X_tfidf_sentiment, y_sentiment, test_size=0.2, random_state=42)

# Train a Random Forest Classifier for sentiment prediction
sentiment_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=18)
sentiment_model.fit(X_train_sentiment, y_train_sentiment)

# Predict sentiment labels on the test set
y_pred_sentiment = sentiment_model.predict(X_test_sentiment)

# Evaluate sentiment classifier
print("Sentiment Classification Report:")
print(classification_report(y_test_sentiment, y_pred_sentiment,zero_division=1))
print(f"Accuracy: {accuracy_score(y_test_sentiment, y_pred_sentiment):.2f}")

# Use the entire dataset to predict sentiment labels (since this will be used for the regression model)
upsampled['Predicted_Sentiment'] = sentiment_model.predict(X_tfidf_sentiment)

# Features and target selection for IMDB rating prediction
X_rating = upsampled['reviewText']
y_rating = upsampled['Rating']

# Step 1: Convert raw text to a count matrix using the same CountVectorizer
X_counts_rating = count_vectorizer.transform(X_rating)

# Step 2: Transform the count matrix to a TF-IDF representation using the same TfidfTransformer
X_tfidf_rating = tfidf_transformer_sentiment.transform(X_counts_rating)

# Step 3: Combine TF-IDF features with predicted sentiment labels as an additional feature
# Convert to DataFrame for easier manipulation
import scipy.sparse as sp
predicted_sentiment_features = sp.csr_matrix(upsampled['Predicted_Sentiment'].values).T  # Convert to sparse matrix
X_combined = sp.hstack([X_tfidf_rating, predicted_sentiment_features])  # Combine TF-IDF and sentiment features

# Train-test split for IMDB rating prediction
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X_combined, y_rating, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor for IMDB rating prediction
model = LinearRegression()
model.fit(X_train_rating, y_train_rating)

# Ennustetaan testidatalla
y_pred_rating = model.predict(X_test_rating)


# Calculate and display metrics for IMDB rating prediction
mse = mean_squared_error(y_test_rating, y_pred_rating)
r2 = r2_score(y_test_rating, y_pred_rating)
print(f"Mean Squared Error (IMDB Rating Prediction): {mse:.2f}")
print(f"R-squared (IMDB Rating Prediction): {r2:.2f}")

# Save both models and transformers to files using pickle
with open('sentiment_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(sentiment_model, model_file)

with open('random_forest_regressor_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('count_vectorizer.pkl', 'wb') as count_vectorizer_file:
    pickle.dump(count_vectorizer, count_vectorizer_file)

with open('tfidf_transformer.pkl', 'wb') as tfidf_transformer_file:
    pickle.dump(tfidf_transformer_sentiment, tfidf_transformer_file)

print("Models and transformers have been pickled and saved to files.")

def test_review(review_text):
    # Transform the review text using the loaded count vectorizer and TF-IDF transformer
    X_count_review = count_vectorizer.transform([review_text])
    X_tfidf_review = tfidf_transformer_sentiment.transform(X_count_review)

    # Predict the sentiment of the review
    predicted_sentiment = sentiment_model.predict(X_tfidf_review)[0]

    # Combine the TF-IDF features with the predicted sentiment as input to the IMDB model
    predicted_sentiment_feature = sp.csr_matrix([predicted_sentiment])  # Convert to sparse matrix
    X_combined_review = sp.hstack([X_tfidf_review, predicted_sentiment_feature])

    # Predict the IMDB rating using the combined features
    predicted_rating = model.predict(X_combined_review)[0]

    # Print results
    sentiment_label = 'Positive' if predicted_sentiment == 1 else 'Negative'
    print(f"Review: {review_text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted IMDB Rating: {predicted_rating:.2f}")

# Test example reviews
print("\nTesting example reviews...\n")

example_reviews = [
    "This movie was absolutely fantastic! The plot was gripping and the performances were outstanding.",
    "I really disliked this movie. The storyline was predictable and the acting was subpar.",
    "An average movie with some good and some bad moments. Not the best but not the worst either.",
    "The movie exceeded all my expectations. Brilliant direction and a must-watch for everyone.",
    "shit ass poop disgusting nauseating bad terrible",
    "amazing outstanding best good cool nice brilliant great love"
]

for review in example_reviews:
    test_review(review)
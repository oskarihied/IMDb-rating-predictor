# Neural Network using Scikit-learn's MLPClassifier and MLPRegressor
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy.sparse as sp

# Load the dataset
print("Loading data...")
final_df = pd.read_csv('combined.csv')

# Create sentiment labels based on sentiment score
print("Creating sentiment labels...")
final_df['Sentiment_Label'] = final_df['scoreSentiment'].apply(lambda x: 1 if x == "POSITIVE" else 0)

# Upsampling the minority class for sentiment classification
print("Upsampling the minority class for sentiment...")
majority_class = final_df[final_df['Sentiment_Label'] == 1]
minority_class = final_df[final_df['Sentiment_Label'] == 0]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
upsampled = pd.concat([majority_class, minority_upsampled])
print("Upsampling completed.")

# Define rating categories based on numerical rating values
def categorize_rating(rating):
    if rating <= 4:
        return 'very_low'
    elif rating <= 6:
        return 'low'
    elif rating <= 8:
        return 'medium'
    else:
        return 'high'

print("Categorizing ratings...")
upsampled['Rating_Category'] = upsampled['Rating'].apply(categorize_rating)

# Upsample each rating category to match the most common category size
print("Upsampling each rating category...")
max_category_count = upsampled['Rating_Category'].value_counts().max()
upsampled_categories = []
for category in ['very_low', 'low', 'medium', 'high']:
    category_data = upsampled[upsampled['Rating_Category'] == category]
    upsampled_category = resample(category_data, replace=True, n_samples=max_category_count, random_state=42)
    upsampled_categories.append(upsampled_category)
upsampled = pd.concat(upsampled_categories)
print("Upsampling for rating categories completed.")

# Features and labels for sentiment classification
X_sentiment = upsampled['reviewText']
y_sentiment = upsampled['Sentiment_Label']

# Use HashingVectorizer to convert text data into numerical format
print("Vectorizing sentiment text data...")
hashing_vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
X_hashed_sentiment = hashing_vectorizer.fit_transform(X_sentiment)

# Train-test split for sentiment classification
print("Splitting data into training and testing sets for sentiment classification...")
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
    X_hashed_sentiment, y_sentiment, test_size=0.2, random_state=42)
print(f"Training data size: {X_train_sentiment.shape[0]}, Test data size: {X_test_sentiment.shape[0]}")

# Build and train the sentiment classifier model with verbose logging
print("Building and training the sentiment classification model...")
sentiment_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=100, random_state=42, verbose=True)
sentiment_model.fit(X_train_sentiment, y_train_sentiment)

# Predict and evaluate sentiment classifier
print("Predicting sentiment on test set...")
y_pred_sentiment = sentiment_model.predict(X_test_sentiment)
print("Evaluating sentiment classification performance...")
print("Sentiment Classification Report:")
print(classification_report(y_test_sentiment, y_pred_sentiment, zero_division=1))
print(f"Accuracy: {accuracy_score(y_test_sentiment, y_pred_sentiment):.2f}")

# Confusion Matrix for sentiment classification
cm = confusion_matrix(y_test_sentiment, y_pred_sentiment)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])

# Plot confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Sentiment Classification')
plt.show()

# Use the entire dataset to predict sentiment labels
print("Predicting sentiment labels for the entire dataset...")
upsampled['Predicted_Sentiment'] = sentiment_model.predict(X_hashed_sentiment)

# Features and target for IMDB rating prediction
X_rating = upsampled['reviewText']
y_rating = upsampled['Rating']
X_hashed_rating = hashing_vectorizer.transform(X_rating)

# Combine hashed features with predicted sentiment
print("Combining hashed features with predicted sentiment...")
X_combined = sp.hstack([X_hashed_rating, sp.csr_matrix(upsampled['Predicted_Sentiment'].values).T])

# Train-test split for rating prediction
print("Splitting data into training and testing sets for rating prediction...")
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X_combined, y_rating, test_size=0.2, random_state=42)
print(f"Training data size: {X_train_rating.shape[0]}, Test data size: {X_test_rating.shape[0]}")

# Build and train the rating prediction model
print("Building and training the rating prediction model...")
rating_model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=100, random_state=42, verbose=True)
rating_model.fit(X_train_rating, y_train_rating)

# Predict on the test set
print("Predicting IMDB ratings on test set...")
y_pred_rating = rating_model.predict(X_test_rating)

# Evaluate the model
print("Evaluating rating prediction performance...")
mse = mean_squared_error(y_test_rating, y_pred_rating)
r2 = r2_score(y_test_rating, y_pred_rating)
print(f"Mean Squared Error (IMDB Rating Prediction): {mse:.2f}")
print(f"R-squared (IMDB Rating Prediction): {r2:.2f}")

# Plotting Actual vs Predicted IMDB Ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rating, y_pred_rating, color='blue', alpha=0.6, label='Actual vs Predicted')
plt.plot([min(y_test_rating), max(y_test_rating)], [min(y_test_rating), max(y_test_rating)], color='red', linewidth=2, label='Ideal Fit Line')
plt.title('Neural Network: Actual vs Predicted IMDB Ratings')
plt.xlabel('Actual IMDB Rating')
plt.ylabel('Predicted IMDB Rating')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for trend line visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rating, y_pred_rating, color='blue', alpha=0.6, label='Actual Data')
plt.plot(sorted(y_test_rating), sorted(y_pred_rating), color='orange', linewidth=2, label='Trend Line')
plt.title('Predicted IMDB Rating Trend Line')
plt.xlabel('Actual IMDB Rating')
plt.ylabel('Predicted IMDB Rating')
plt.legend()
plt.grid(True)
plt.show()

print("Completed all tasks.")

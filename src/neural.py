#neural network
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import HashingVectorizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import pickle
import scipy.sparse as sp

# Load the data
final_df = pd.read_csv('combined.csv')

# Create sentiment labels
final_df['Sentiment_Label'] = final_df['scoreSentiment'].apply(lambda x: 1 if x == "POSITIVE" else 0)

# Upsample the minority class for sentiment
majority_class = final_df[final_df['Sentiment_Label'] == 1]
minority_class = final_df[final_df['Sentiment_Label'] == 0]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
upsampled = pd.concat([majority_class, minority_upsampled])


# Define rating categories
def categorize_rating(rating):
    if rating <= 4:
        return 'very_low'
    elif rating <= 6:
        return 'low'
    elif rating <= 8:
        return 'medium'
    else:
        return 'high'


upsampled['Rating_Category'] = upsampled['Rating'].apply(categorize_rating)

# Get the count of the most common category
max_category_count = upsampled['Rating_Category'].value_counts().max()

# Upsample each category to match the most common category
upsampled_categories = []
for category in ['very_low', 'low', 'medium', 'high']:
    category_data = upsampled[upsampled['Rating_Category'] == category]
    upsampled_category = resample(category_data,
                                  replace=True,
                                  n_samples=max_category_count,
                                  random_state=42)
    upsampled_categories.append(upsampled_category)

# Combine all upsampled categories
upsampled = pd.concat(upsampled_categories)

print(f"Total samples after upsampling: {len(upsampled)}")
print(f"Positive sentiment samples: {len(upsampled[upsampled['Sentiment_Label'] == 1])}")
print(f"Negative sentiment samples: {len(upsampled[upsampled['Sentiment_Label'] == 0])}")
print(upsampled['Rating_Category'].value_counts())

# Features and labels
X_sentiment = upsampled['reviewText']
y_sentiment = upsampled['Sentiment_Label']

# Use HashingVectorizer
hashing_vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
X_hashed_sentiment = hashing_vectorizer.fit_transform(X_sentiment)

# Train-test split for sentiment classification
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
    X_hashed_sentiment, y_sentiment, test_size=0.2, random_state=42)


# Custom layer to handle sparse input
class SparseInput(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SparseInput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]), self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SparseInput, self).build(input_shape)

    def call(self, x):
        return tf.sparse.sparse_dense_matmul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# Build the Neural Network model for sentiment classification
input_layer = layers.Input(shape=(X_train_sentiment.shape[1],), sparse=True)
x = SparseInput(64)(input_layer)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_sentiment, y_train_sentiment, epochs=5, batch_size=64, validation_split=0.2)

# Predict and evaluate sentiment classifier
y_pred_sentiment = (model.predict(X_test_sentiment) > 0.5).astype("int32")
print("Sentiment Classification Report:")
print(classification_report(y_test_sentiment, y_pred_sentiment, zero_division=1))
print(f"Accuracy: {accuracy_score(y_test_sentiment, y_pred_sentiment):.2f}")

# Use the entire dataset to predict sentiment labels
upsampled['Predicted_Sentiment'] = (model.predict(X_hashed_sentiment) > 0.5).astype("int32")

# Features and target selection for IMDB rating prediction
X_rating = upsampled['reviewText']
y_rating = upsampled['Rating']

# Transform the ratings using HashingVectorizer
X_hashed_rating = hashing_vectorizer.transform(X_rating)

# Combine hashed features with predicted sentiment labels
X_combined = sp.hstack([X_hashed_rating, sp.csr_matrix(upsampled['Predicted_Sentiment'].values).T])

# Train-test split for IMDB rating prediction
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X_combined, y_rating, test_size=0.2, random_state=42)

# Build the Neural Network model for IMDB Rating prediction
input_layer = layers.Input(shape=(X_train_rating.shape[1],), sparse=True)
x = SparseInput(64)(input_layer)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

rating_model = Model(inputs=input_layer, outputs=output)

# Compile and train the model for IMDB rating prediction
rating_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
rating_model.fit(X_train_rating, y_train_rating, epochs=5, batch_size=64, validation_split=0.2)

# Predict ratings
y_pred_rating = rating_model.predict(X_test_rating)

# Calculate metrics for IMDB rating prediction
mse = mean_squared_error(y_test_rating, y_pred_rating)
r2 = r2_score(y_test_rating, y_pred_rating)
print(f"Mean Squared Error (IMDB Rating Prediction): {mse:.2f}")
print(f"R-squared (IMDB Rating Prediction): {r2:.2f}")

# Save models and vectorizer
model.save('sentiment_classifier_model.h5')
rating_model.save('imdb_rating_model.h5')
with open('hashing_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(hashing_vectorizer, vectorizer_file)

print("Models and vectorizer have been saved.")


# Function to test reviews
def test_review(review_text):
    X_hashed_review = hashing_vectorizer.transform([review_text])

    # Predict sentiment
    predicted_sentiment = (model.predict(X_hashed_review) > 0.5).astype("int32")[0][0]

    # Combine the hashed features with the predicted sentiment
    X_combined_review = sp.hstack([X_hashed_review, sp.csr_matrix([[predicted_sentiment]])])

    # Predict the IMDB rating
    predicted_rating = rating_model.predict(X_combined_review)[0][0]

    # Print results
    sentiment_label = 'Positive' if predicted_sentiment == 1 else 'Negative'
    print(f"Review: {review_text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted IMDB Rating: {predicted_rating:.2f}")


# Test example reviews
sample_reviews = [
    "An outstanding movie with great acting and direction.",
    "The movie was too slow and lacked substance.",
    "A brilliant performance but the storyline was predictable.",
    "This movie was absolutely terrible. Worst film I've ever seen.",
    "A mediocre film, neither good nor bad.",
    "shit ass poop disgusting nauseating bad terrible",
    "amazing outstanding best good cool nice brilliant"
]

# Testing each review
for review in sample_reviews:
    test_review(review)
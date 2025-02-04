#neural network
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import HashingVectorizer
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Upsample each rating category to the most common category size
max_category_count = upsampled['Rating_Category'].value_counts().max()
upsampled_categories = []
for category in ['very_low', 'low', 'medium', 'high']:
    category_data = upsampled[upsampled['Rating_Category'] == category]
    upsampled_category = resample(category_data, replace=True, n_samples=max_category_count, random_state=42)
    upsampled_categories.append(upsampled_category)
upsampled = pd.concat(upsampled_categories)

# Features and labels
X_sentiment = upsampled['reviewText']
y_sentiment = upsampled['Sentiment_Label']

# Use HashingVectorizer
hashing_vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
X_hashed_sentiment = hashing_vectorizer.fit_transform(X_sentiment)

# Train-test split for sentiment classification
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
    X_hashed_sentiment, y_sentiment, test_size=0.2, random_state=42)

# Custom sparse input layer
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

# Build and train the sentiment classifier model
input_layer = layers.Input(shape=(X_train_sentiment.shape[1],), sparse=True)
x = SparseInput(64)(input_layer)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)
sentiment_model = Model(inputs=input_layer, outputs=output)

sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
sentiment_model.fit(X_train_sentiment, y_train_sentiment, epochs=5, batch_size=64, validation_split=0.2)

# Predict and evaluate sentiment classifier
y_pred_sentiment = (sentiment_model.predict(X_test_sentiment) > 0.5).astype("int32")
print("Sentiment Classification Report:")
print(classification_report(y_test_sentiment, y_pred_sentiment, zero_division=1))
print(f"Accuracy: {accuracy_score(y_test_sentiment, y_pred_sentiment):.2f}")

# Plotting Sentiment Classification Accuracy
# Confusion Matrix for sentiment classification
cm = confusion_matrix(y_test_sentiment, y_pred_sentiment)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])

# Plot confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Sentiment Classification')
plt.show()

# Use the entire dataset to predict sentiment labels
upsampled['Predicted_Sentiment'] = (sentiment_model.predict(X_hashed_sentiment) > 0.5).astype("int32")

# Features and target for IMDB rating prediction
X_rating = upsampled['reviewText']
y_rating = upsampled['Rating']
X_hashed_rating = hashing_vectorizer.transform(X_rating)
# Combine hashed features with predicted sentiment
X_combined = sp.hstack([X_hashed_rating, sp.csr_matrix(upsampled['Predicted_Sentiment'].values).T])

# Train-test split for rating prediction
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X_combined, y_rating, test_size=0.2, random_state=42)

# Build and train the rating prediction model
input_layer = layers.Input(shape=(X_train_rating.shape[1],), sparse=True)
x = SparseInput(64)(input_layer)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)
rating_model = Model(inputs=input_layer, outputs=output)

rating_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
rating_model.fit(X_train_rating, y_train_rating, epochs=5, batch_size=64, validation_split=0.2)

# Predict on the test set
y_pred_rating = rating_model.predict(X_test_rating)
y_pred_train = rating_model.predict(X_train_rating)

# Evaluate the model
mse = mean_squared_error(y_test_rating, y_pred_rating)
mse2 = mean_squared_error(y_train_rating, y_pred_train)
r2 = r2_score(y_test_rating, y_pred_rating)
print(f"Mean Squared Error (IMDB Rating Prediction): {mse:.2f}")
print(f"train: {mse2:.2f}")
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
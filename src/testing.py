
#Only for testing the neural network model

import pickle
import scipy.sparse as sp
import tensorflow as tf
from tensorflow import keras
from keras import layers

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


# Load the models and vectorizer
sentiment_model = keras.models.load_model('sentiment_classifier_model.h5', custom_objects={'SparseInput': SparseInput})
rating_model = keras.models.load_model('imdb_rating_model.h5', custom_objects={'SparseInput': SparseInput})

with open('hashing_vectorizer.pkl', 'rb') as vectorizer_file:
    hashing_vectorizer = pickle.load(vectorizer_file)

def test_review(review_text):
    X_hashed_review = hashing_vectorizer.transform([review_text])
    predicted_sentiment = (sentiment_model.predict(X_hashed_review) > 0.5).astype("int32")[0][0]
    X_combined_review = sp.hstack([X_hashed_review, sp.csr_matrix([[predicted_sentiment]])])
    predicted_rating = rating_model.predict(X_combined_review)[0][0]

    sentiment_label = 'Positive' if predicted_sentiment == 1 else 'Negative'
    print(f"Review: {review_text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted IMDB Rating: {predicted_rating:.2f}")

# Sample reviews to test
sample_reviews = [
    "An outstanding movie with great acting and direction.",
    "The movie was too slow and lacked substance.",
    "A brilliant performance but the storyline was predictable.",
    "This movie was absolutely terrible. Worst film I've ever seen.",
    "A mediocre film, neither good nor bad.",
    "shit ass poop disgusting nauseating bad terrible",
    "amazing outstanding best good cool nice brilliant",
    """When you watch a movie that you don't understand the meaning of, a movie that after watching it you basically think is useless. This is the sequel to Joker. I honestly expected something much better, I'm disappointed, I can't understand the reason for the movie, there's almost nothing that surprised me. Maybe the only feeling I really felt during the movie was boredom. I wonder if there was really a need for a Joker sequel, in my opinion no, in fact this disappointing sequel confirms my opinion. Sometimes it's better to stick with one film rather than have disappointing sequels like this film.""",
    "It is just what you want for the best movie. Great story great acting, thrilling twist. Just watched Joker in 2019, I just has to come back and give dark knight a 10. And thanks to Heath Ledger for the exceptional performs.",
    "Best movie ever. Heath ledger's work is phenomenal no words.....",
    "Shawshank redemption",
    ""
]

# Testing each review
for review in sample_reviews:
    test_review(review)
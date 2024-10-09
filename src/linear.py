#linear
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

X = final_df['reviewText']
y = final_df['Rating']


# Muutetaan tekstidata numeeriseen muotoon käyttäen TfidfVectorizeria
vectorizer = TfidfVectorizer(max_features=30000, stop_words='english',max_df=0.5)  # Käytetään 5000 yleisintä sanaa
X_transformed = vectorizer.fit_transform(X)

# Jaetaan data koulutus- ja testidataan (80% koulutukseen ja 20% testiin)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Luodaan ja koulutetaan lineaarinen regressiomalli
model = LinearRegression()
model.fit(X_train, y_train)

# Ennustetaan testidatalla
y_pred = model.predict(X_test)

# Lasketaan mallin suoriutumismittarit
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Testataan muutamalla esimerkkitapauksella
sample_reviews = [
    "An outstanding movie with great acting and direction.",
    "The movie was too slow and lacked substance.",
    "A brilliant performance but the storyline was predictable.",
    "shit ass poop disgusting nauseating bad terrible",
    "amazing outstanding best good cool nice brilliant"
]
sample_transformed = vectorizer.transform(sample_reviews)
sample_predictions = model.predict(sample_transformed)

for review, prediction in zip(sample_reviews, sample_predictions):
    print(f"Review: {review}")
    print(f"Predicted IMDB Rating: {prediction:.2f}\n")
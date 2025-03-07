import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    # Load the dataset
    df = pd.read_csv('../data/windows_commands_dataset.csv')

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['user_input'], df['command'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer
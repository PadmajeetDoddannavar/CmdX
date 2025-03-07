import joblib
import os

def save_model(model, vectorizer):
    # Create the models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)

    # Save the model and vectorizer
    joblib.dump(model, '../models/command_model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')

    print("Model and vectorizer saved successfully.")
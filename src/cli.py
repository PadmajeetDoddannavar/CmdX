import joblib
import subprocess
import os

def load_model():
    model_path = '../models/command_model.pkl'
    vectorizer_path = '../models/vectorizer.pkl'

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        exit(1)

    if not os.path.exists(vectorizer_path):
        print(f"Error: Vectorizer file '{vectorizer_path}' not found. Please train the model first.")
        exit(1)

    # Load the model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def execute_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def main():
    model, vectorizer = load_model()

    while True:
        user_input = input("What would you like to do? (type 'exit' to quit) ")
        if user_input.lower() == 'exit':
            break

        user_input_vectorized = vectorizer.transform([user_input])
        command = model.predict(user_input_vectorized)[0]

        confirm = input(f"Do you want to execute: {command}? (yes/no) ")
        if confirm.lower() == 'yes':
            output = execute_command(command)
            print(output)

if __name__ == "__main__":
    main()
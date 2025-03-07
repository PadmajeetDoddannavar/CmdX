from train_model import train_model
from save_model import save_model

def main():
    model, vectorizer = train_model()
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
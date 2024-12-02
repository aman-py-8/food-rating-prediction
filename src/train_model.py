import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(train_path):
    """
    Load the preprocessed training dataset from the provided path.
    """
    train_df = pd.read_csv(train_path)
    return train_df

def preprocess_features_target(df, target_column):
    """
    Split the dataset into features and target.
    """
    features = df.drop(target_column, axis=1)
    target = df[target_column]
    return features, target

def vectorize_text(features, text_column):
    """
    Vectorize text features using TF-IDF.
    """
    tfidf = TfidfVectorizer()
    features_vectorized = tfidf.fit_transform(features[text_column])
    return features_vectorized, tfidf

def train_model(X_train, y_train):
    """
    Train a RandomForest model on the training data.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset and print metrics.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

def save_model(model, tfidf, model_path, vectorizer_path):
    """
    Save the trained model and TF-IDF vectorizer to disk.
    """
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    # File paths
    train_path = "data/preprocessed_train.csv"  # Use preprocessed data
    model_path = "models/food_rating_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    # Load the preprocessed data
    print("Loading data...")
    train_df = load_data(train_path)

    # Preprocess features and target
    print("Preprocessing features and target...")
    target_column = "Rating"  # Replace with the actual target column name
    text_column = "Recipe_Review"  # Replace with the actual text column name
    features, target = preprocess_features_target(train_df, target_column)

    # Vectorize text data
    print("Vectorizing text data...")
    X_vectorized, tfidf = vectorize_text(features, text_column)

    # Split into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, target, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model and vectorizer
    print("Saving the model and vectorizer...")
    save_model(model, tfidf, model_path, vectorizer_path)

    print("Training complete!")

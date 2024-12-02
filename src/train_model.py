import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os  # To create directories if they do not exist
import matplotlib.pyplot as plt
import seaborn as sns

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
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Classification Report:\n", class_report)
    print("Accuracy Score:", accuracy)
    
    return class_report, accuracy

def save_model(model, tfidf, model_path, vectorizer_path):
    """
    Save the trained model and TF-IDF vectorizer to disk.
    """
    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def plot_confusion_matrix(y_test, y_pred, results_dir):
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot in the results folder
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.close()

def save_results(class_report, accuracy, results_dir):
    """
    Save the evaluation results (classification report, accuracy score) to the results folder.
    """
    # Ensure the 'results' directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save classification report to a text file
    with open(f"{results_dir}/classification_report.txt", "w") as f:
        f.write(class_report)
    
    # Save accuracy score to a text file
    with open(f"{results_dir}/accuracy_score.txt", "w") as f:
        f.write(f"Accuracy Score: {accuracy}\n")

if __name__ == "__main__":
    # File paths
    train_path = "data/preprocessed_train.csv"  # Use preprocessed data
    model_path = "models/food_rating_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    results_dir = "results"  # Folder to store evaluation results

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
    class_report, accuracy = evaluate_model(model, X_test, y_test)

    # Save the results (classification report, accuracy score)
    print("Saving evaluation results...")
    save_results(class_report, accuracy, results_dir)

    print("Saving confusion matrix plot...")
    plot_confusion_matrix(y_test, model.predict(X_test), results_dir)

    # Save the model and vectorizer
    print("Saving the model and vectorizer...")
    save_model(model, tfidf, model_path, vectorizer_path)

    print("Training complete!")

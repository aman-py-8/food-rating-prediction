import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path):
    """
    Load training and test datasets from the provided paths.
    """
    logging.info(f"Loading data from {train_path} and {test_path}...")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    return train_df, test_df

def separate_features_target(df, target_column):
    """
    Separate the features and target column from the dataset.
    """
    logging.info(f"Separating features and target column '{target_column}'...")
    target = df[target_column]
    features = df.drop(target_column, axis=1)
    logging.info(f"Features List: {list(features.columns)}")
    logging.info(f"Total Features: {len(features.columns)}")
    return features, target

def remove_outliers(df, column_name, threshold=10):
    """
    Detect and remove outliers in a specific column using Z-scores.
    """
    logging.info(f"Removing outliers from column '{column_name}' using Z-scores...")
    z_scores = np.abs(stats.zscore(df[column_name]))
    outlier_indices = np.where(z_scores > threshold)[0]
    logging.info(f"Found {len(outlier_indices)} outliers in column {column_name}. Removing them...")
    df_cleaned = df.drop(outlier_indices, axis=0).reset_index(drop=True)
    return df_cleaned

def preprocess_data(train_df, test_df, target_column):
    """
    Preprocess the training and test datasets:
    - Handle missing values
    - Detect and remove outliers
    """
    # Separate features and target from the train dataset
    logging.info(f"Preprocessing data for target column '{target_column}'...")
    train_features, train_target = separate_features_target(train_df, target_column)
    
    # Create imputers for numeric and categorical columns
    logging.info("Creating imputers for numeric and categorical columns...")
    numeric_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    # Identify numeric and categorical columns
    numeric_columns = train_features.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = train_features.select_dtypes(include=["object"]).columns

    # Apply imputer to numeric columns
    logging.info(f"Imputing missing values in numeric columns: {list(numeric_columns)}...")
    train_df[numeric_columns] = numeric_imputer.fit_transform(train_df[numeric_columns])
    test_df[numeric_columns] = numeric_imputer.transform(test_df[numeric_columns])

    # Apply imputer to categorical columns
    logging.info(f"Imputing missing values in categorical columns: {list(categorical_columns)}...")
    train_df[categorical_columns] = categorical_imputer.fit_transform(train_df[categorical_columns])
    test_df[categorical_columns] = categorical_imputer.transform(test_df[categorical_columns])

    # Example: Remove outliers from a specific column in the training dataset
    logging.info("Removing outliers from the 'UserReputation' column...")
    train_df = remove_outliers(train_df, "UserReputation", threshold=10)

    logging.info("Preprocessing complete.")
    return train_df, test_df

def save_preprocessed_data(train_df, test_df, train_output_path, test_output_path):
    """
    Save the preprocessed train and test data to CSV files.
    """
    logging.info(f"Saving preprocessed data to {train_output_path} and {test_output_path}...")
    try:
        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)
        logging.info(f"Preprocessed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")
        raise


if __name__ == "__main__":
    # File paths
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_output_path = "data/preprocessed_train.csv"
    test_output_path = "data/preprocessed_test.csv"

    # Load data
    logging.info("Starting preprocessing...")
    train_df, test_df = load_data(train_path, test_path)

    # Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df, target_column="Rating")

    # Save the preprocessed data
    save_preprocessed_data(train_df, test_df, train_output_path, test_output_path)

    logging.info("Preprocessing complete.")

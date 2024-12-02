import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import plot_kde, describe_data

def load_data(file_path):
    """
    Load dataset from the given file path.
    """
    return pd.read_csv(file_path)

def plot_missing_values(df):
    """
    Visualize the missing values in the dataset.
    """
    print("Visualizing missing values...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_feature_distribution(df, feature):
    """
    Plot the distribution of a feature using a histogram and KDE.
    """
    print(f"Plotting distribution for {feature}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot a heatmap of the correlation matrix for the numerical features.
    """
    print("Plotting correlation matrix...")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def eda(df):
    """
    Perform EDA on the dataset:
    - Summary of dataset
    - Missing values visualization
    - Feature distribution plotting
    - Correlation matrix
    """
    # Dataset summary
    describe_data(df)

    # Visualizing missing values
    plot_missing_values(df)

    # Plot distributions for relevant numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        plot_feature_distribution(df, column)

    # Plot correlation matrix for numerical features
    plot_correlation_matrix(df)

if __name__ == "__main__":
    # Path to your dataset
    data_path = 'data/train.csv'  # Update with actual path

    # Load the data
    print("Loading dataset...")
    df = load_data(data_path)

    # Perform EDA
    print("Performing EDA...")
    eda(df)

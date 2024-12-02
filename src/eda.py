import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import describe_data

def load_data(file_path):
    """
    Load dataset from the given file path.
    """
    return pd.read_csv(file_path)

def save_plots_to_pdf(plots, output_file):
    """
    Save all the plots to a single PDF file.
    """
    with PdfPages(output_file) as pdf:
        for fig in plots:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"All plots saved to {output_file}")

def plot_missing_values(df):
    """
    Visualize the missing values in the dataset and return the plot.
    """
    print("Visualizing missing values...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    return plt.gcf()

def plot_feature_distribution(df, feature):
    """
    Plot the distribution of a feature and return the plot.
    """
    print(f"Plotting distribution for {feature}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    return plt.gcf()

def plot_rating_distribution(df, target_column):
    """
    Plot the distribution of the target variable and return the plot.
    """
    print("Plotting distribution of the target variable...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=target_column, palette='viridis')
    plt.title('Rating Distribution')
    return plt.gcf()

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix and return the plot.
    """
    print("Plotting correlation matrix...")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    return plt.gcf()

def eda(df, pdf_output_path):
    """
    Perform EDA on the dataset:
    - Summary of dataset
    - Missing values visualization
    - Feature distribution plotting
    - Correlation matrix
    - Rating distribution
    """
    # Dataset summary
    describe_data(df)

    # Prepare a list to store all the figures
    figures = []

    # Visualizing missing values
    figures.append(plot_missing_values(df))

    # Plot distributions for relevant numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        figures.append(plot_feature_distribution(df, column))

    # Plot Rating distribution
    figures.append(plot_rating_distribution(df, target_column="Rating"))

    # Plot correlation matrix for numerical features
    figures.append(plot_correlation_matrix(df))

    # Save all figures to a single PDF
    save_plots_to_pdf(figures, pdf_output_path)

if __name__ == "__main__":
    # Path to your dataset
    data_path = 'data/train.csv'  # Update with actual path
    pdf_output_path = 'results/eda_plots.pdf'

    # Load the data
    print("Loading dataset...")
    df = load_data(data_path)

    # Perform EDA
    print("Performing EDA...")
    eda(df, pdf_output_path)

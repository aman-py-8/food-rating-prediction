import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_histograms(df, figsize=(30, 24)):
    """
    Plot histograms for all columns in the dataset.
    """
    print("Plotting histograms for all columns...")
    df.hist(figsize=figsize)
    plt.show()

def plot_kde(df, column):
    """
    Plot a Kernel Density Estimation (KDE) for a specified column.
    """
    print(f"Plotting KDE plot for {column}...")
    sns.kdeplot(df[column], fill=True)
    plt.title(f'Kernel Density Estimation (KDE) Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

def plot_boxplot(df, column):
    """
    Plot a boxplot for a specific column.
    """
    print(f"Plotting box plot for {column}...")
    plt.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel('Values')
    plt.show()

def get_quarter(month):
    """
    Map a month number to its respective quarter.
    """
    if month in range(1, 4):
        return 1
    elif month in range(4, 7):
        return 2
    elif month in range(7, 10):
        return 3
    elif month in range(10, 13):
        return 4
    else:
        raise ValueError("Month must be in the range 1-12")

def describe_data(df):
    """
    Print a summary of the dataset, including missing values and data types.
    """
    print("Dataset Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())

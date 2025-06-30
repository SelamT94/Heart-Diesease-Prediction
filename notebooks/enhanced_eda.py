# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for better visuals
sns.set_style('whitegrid')

# %%
# Define file paths
RAW_DATA_PATH = '../data/raw/heart.csv'

# %%
# Load the raw data
try:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print("✅ Raw data loaded successfully.")
    print(f"Dataset contains {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")
except FileNotFoundError:
    print(f"❌ Error: The file was not found at {RAW_DATA_PATH}")
    print("Please ensure the dataset is in the 'data/raw/' directory.")

# %%
# Display basic information
if 'df_raw' in locals():
    print("\nFirst 5 rows of the raw dataset:")
    print(df_raw.head())

    print("\nDataset Column Types and Non-Null Counts:")
    df_raw.info()

    print("\nStatistical summary of numerical features:")
    print(df_raw.describe())

# %%
# Check for missing values
if 'df_raw' in locals():
    print("\nChecking for missing values:")
    missing_values = df_raw.isnull().sum()
    print(missing_values[missing_values > 0])

# %%
# Binarize Target Variable
if 'df_raw' in locals():
    # The 'num' column indicates the degree of heart disease, from 0 (no disease) to 4.
    # For a simpler classification model, we convert this into a binary problem.
    # 0 = No heart disease, 1 = Heart disease present
    df_raw['target'] = (df_raw['num'] > 0).astype(int)

    # Visualize the distribution of the new binary target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df_raw)
    plt.title('Distribution of Binary Heart Disease Target')
    plt.xlabel('0 = No Heart Disease, 1 = Has Heart Disease')
    plt.ylabel('Patient Count')

    # Adding annotations for clarity
    total = len(df_raw['target'])
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5, f'{height}\n({height/total:.1%})', ha="center")
    plt.show()

# %%
# Analyze Numerical Features
if 'df_raw' in locals():
    # Age distribution with the new binary target.
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_raw, x='age', hue='target', kde=True, bins=30, palette='viridis')
    plt.title('Age Distribution by Heart Disease Status')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(title='Heart Disease', labels=['Yes', 'No'])
    plt.show()

    # Boxplots for other numerical features
    numerical_features = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='target', y=feature, data=df_raw, palette='viridis')
        plt.title(f'{feature.capitalize()} by Heart Disease Status')
        plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
        plt.ylabel(feature.capitalize())
        plt.show()

# %%
# Analyze Categorical Features
if 'df_raw' in locals():
    # Convert boolean-like object columns to a numerical format.
    for col in ['fbs', 'exang']:
        if col in df_raw.columns and df_raw[col].dtype == 'object':
            print(f"Unique values in '{col}' before conversion: {df_raw[col].unique()}")
            # Using a map is safer and handles NaNs gracefully.
            df_raw[col] = df_raw[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
            print(f"Unique values in '{col}' after conversion: {df_raw[col].unique()}")
            print("-" * 30)

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'dataset']

    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue='target', data=df_raw, palette='viridis')
        plt.title(f'Distribution of {feature} by Heart Disease Status')
        plt.xlabel(feature)
        plt.ylabel('Patient Count')
        plt.xticks(rotation=45)
        plt.legend(title='Heart Disease', labels=['No', 'Yes'])
        plt.show()


# %%
# Correlation Analysis
if 'df_raw' in locals():
    # Compute the correlation matrix for numerical features and the binary target.
    plt.figure(figsize=(12, 10))
    # Select the original numerical features plus the new 'target'
    corr_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'target']
    correlation_matrix = df_raw[corr_features].corr()

    # Visualize the correlation matrix using a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features and Target')
    plt.show() 
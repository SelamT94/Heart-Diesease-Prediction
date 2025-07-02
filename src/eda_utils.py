import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def show_basic_info(df):
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Column Types and Non-Null Counts:")
    df.info()
    print("\nStatistical summary of numerical features:")
    print(df.describe())

def check_missing_values(df):
    print("\nChecking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

def plot_target_distribution(df, target_col='target'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title('Distribution of Binary Heart Disease Target')
    plt.xlabel('0 = No Heart Disease, 1 = Has Heart Disease')
    plt.ylabel('Patient Count')
    total = len(df[target_col])
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5, f'{height}\n({height/total:.1%})', ha="center")
    plt.show()

def plot_numerical_by_target(df, numerical_features, target_col='target'):
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=target_col, y=feature, data=df, palette='viridis')
        plt.title(f'{feature.capitalize()} by Heart Disease Status')
        plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
        plt.ylabel(feature.capitalize())
        plt.show()

def plot_categorical_by_target(df, categorical_features, target_col='target'):
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=target_col, data=df, palette='viridis')
        plt.title(f'Distribution of {feature} by Heart Disease Status')
        plt.xlabel(feature)
        plt.ylabel('Patient Count')
        plt.xticks(rotation=45)
        plt.legend(title='Heart Disease', labels=['No', 'Yes'])
        plt.show()

def plot_correlation_heatmap(df, features):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features and Target')
    plt.show()

def value_counts_categoricals(df, categorical_features):
    for col in categorical_features:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(dropna=False))
        print("-" * 30)

def unique_values_cardinality(df):
    print("\nUnique values and cardinality for each column:")
    for col in df.columns:
        unique_vals = df[col].unique()
        print(f"{col}: {len(unique_vals)} unique values. Sample: {unique_vals[:5]}")

def detect_outliers_iqr(df, numerical_features):
    print("\nOutlier detection using IQR method:")
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers")

def plot_feature_distributions(df, features):
    for col in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def pairplot_features(df, features, target_col='target'):
    sns.pairplot(df[features + [target_col]], hue=target_col, palette='viridis')
    plt.show()

def check_duplicates(df):
    n_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {n_duplicates}")

def zero_negative_summary(df, numerical_features):
    print("\nSummary of zero and negative values:")
    for col in numerical_features:
        n_zero = (df[col] == 0).sum()
        n_neg = (df[col] < 0).sum()
        print(f"{col}: {n_zero} zeros, {n_neg} negatives")

def correlation_with_target(df, target_col='target'):
    print("\nCorrelation of features with target:")
    corrs = df.corr()[target_col].sort_values(ascending=False)
    print(corrs)

def mutual_information_with_target(df, features, target_col='target'):
    """
    Compute mutual information between features and the target.
    Returns a sorted Series of MI scores.
    """
    X = df[features]
    y = df[target_col]
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
    print("\nMutual Information scores with target:")
    print(mi_series)
    return mi_series 
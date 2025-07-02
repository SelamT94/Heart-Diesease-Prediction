import pandas as pd

def drop_missing(df):
    """Drop rows with any missing values."""
    return df.dropna()

def fill_missing(df, fill_value=0):
    """Fill missing values with a specified value (default 0)."""
    return df.fillna(fill_value)

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def convert_dtypes(df, dtype_dict):
    """Convert columns to specified data types using a dictionary."""
    return df.astype(dtype_dict)

def remove_outliers_iqr(df, numerical_features):
    """Remove outliers from numerical features using the IQR method."""
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df 
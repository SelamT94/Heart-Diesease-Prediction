import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif


def binarize_target(df, target_col='num', new_col='target'):
    """Binarize the target column: 0 = no disease, 1 = disease present."""
    df[new_col] = (df[target_col] > 0).astype(int)
    return df


def label_encode_columns(df, columns):
    """Label encode specified categorical columns."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def one_hot_encode_columns(df, columns):
    """One-hot encode specified categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


def scale_numerical_features(df, numerical_features):
    """Standard scale numerical features."""
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def create_interaction_features(df, feature_pairs):
    """Create interaction features (product of pairs)."""
    for (f1, f2) in feature_pairs:
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df


def select_top_features_by_mi(df, features, target_col='target', top_n=10):
    """Select top N features by mutual information with the target."""
    X = df[features]
    y = df[target_col]
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
    top_features = mi_series.head(top_n).index.tolist()
    print(f"Top {top_n} features by mutual information:")
    print(mi_series.head(top_n))
    return top_features 
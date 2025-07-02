import pandas as pd

def load_heart_data(path):
    """
    Loads the heart disease dataset from the specified CSV file path.
    Returns the DataFrame if successful, otherwise returns None.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Data loaded successfully from {path}.")
        print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None 
import pandas as pd

def load_data(path='data/iris.csv'):
    """Load the preprocessed Iris dataset"""
    try:
        df = pd.read_csv(path)
        print(f" Data loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f" File not found at {path}")
        return None

# Example usage
if __name__ == "__main__":
    df = load_data()
    print(df.head())

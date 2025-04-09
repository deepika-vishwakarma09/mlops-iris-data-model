import pandas as pd
from data_injection import load_data

def add_features(df, save_path='data/iris_featurized.csv'):
    # Example Feature: Interaction term
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f" Feature-engineered data saved to {save_path}")
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/iris_preprocessed.csv')  # Load preprocessed data
    add_features(df)

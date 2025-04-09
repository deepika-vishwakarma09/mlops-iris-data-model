import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_injection import load_data

def preprocess_data(df, save_path='data/iris_preprocessed.csv'):
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['target'] = y.reset_index(drop=True)

    # Save to CSV
    df_scaled.to_csv(save_path, index=False)
    print(f"✅ Preprocessed data saved to {save_path}")

    return df_scaled


if __name__ == "__main__":
    df = load_data()  # Load original iris.csv
    preprocess_data(df)  # Now the function is called, and it will print



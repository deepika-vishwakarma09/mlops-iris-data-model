

import pandas as pd
from sklearn.datasets import load_iris

def main():
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target], axis=1)
    df.columns = list(iris.feature_names) + ['target']
    df.to_csv('data/iris.csv', index=False)
    print(" Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    main()

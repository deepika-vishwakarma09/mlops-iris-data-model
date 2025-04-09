import mlflow
import mlflow.sklearn  # For sklearn model logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv('data/iris_featurized.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(max_iter=200),
    "svm": SVC()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)

        model_path = f"model/{name}_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ {name} model trained and saved to {model_path}")

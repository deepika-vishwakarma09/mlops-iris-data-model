import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mlflow

from data_injection import load_data

# Load data
df = load_data('data/iris_featurized.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(max_iter=200),
    "svm": SVC(probability=True)
}

best_model = None
best_score = 0
best_name = ""

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

        if acc > best_score:
            best_model = model
            best_score = acc
            best_name = name

# Save best model
joblib.dump(best_model, "model/best_model.pkl")
print(f"\n🏆 Best model is {best_name} with accuracy {best_score:.4f}")

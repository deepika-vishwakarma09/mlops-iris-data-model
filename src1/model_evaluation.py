import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n {model_name} evaluated successfully!")
    print(f"🔹 Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data/iris_featurized.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_paths = {
        "Random Forest": 'model/random_forest_model.pkl',
        "Logistic Regression": 'model/logistic_regression_model.pkl',
        "SVM": 'model/svm_model.pkl'
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            model = joblib.load(path)
            evaluate_model(model, X_test, y_test, name)
        else:
            print(f" {name} model not found at: {path}")

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Ensure MLFlow is configured to track to a specific experiment
experiment_name = "Hyperparameter_Tuning"
mlflow.set_experiment(experiment_name)

print(f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLFlow Experiment Name: {experiment_name}")


def load_and_prepare_data():
    """Load and prepare the iris dataset"""
    data_path = "data/iris.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop("species", axis=1)
    y_raw = df["species"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_model_and_params(model_name):
    """Get model and hyperparameter grid based on model name"""
    if model_name == "logistic_regression":
        model = LogisticRegression(solver="liblinear", random_state=42)
        param_grid = {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    else:
        raise ValueError(
            "Unsupported model_name. Choose 'logistic_regression' or 'random_forest'."
        )

    return model, param_grid


def evaluate_on_test_set(estimator, X_test, y_test):
    """Evaluate model on test set and return metrics"""
    y_pred = estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    test_precision = precision_score(y_test, y_pred, average="weighted")
    test_recall = recall_score(y_test, y_pred, average="weighted")

    return test_accuracy, test_f1, test_precision, test_recall


def train_model(model_name):
    """
    Trains a model with hyperparameter tuning and logs it to MLflow.

    Args:
        model_name (str): The name of the model to train.
                          Supported values: 'logistic_regression', 'random_forest'.
    """
    print(f"Starting training for model: {model_name}")

    # 1. Prepare Data from iris.csv
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # 2. Define Model and Hyperparameter Grid
    model, param_grid = get_model_and_params(model_name)

    # 3. Perform GridSearchCV with MLFlow Tracking
    model_name_full = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForest",
    }

    with mlflow.start_run(run_name=f"GridSearchCV_Run_{model_name}") as parent_run:
        print(f"MLFlow Parent Run ID: {parent_run.info.run_id}")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("search_strategy", "GridSearchCV")
        mlflow.log_params({"param_grid_" + k: str(v) for k, v in param_grid.items()})

        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Log best parameters and score from GridSearchCV
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", best_score)

        # Evaluate the best model on the test set
        test_accuracy, test_f1, test_precision, test_recall = evaluate_on_test_set(
            best_estimator, X_test, y_test
        )

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)

        print(f"Best parameters: {best_params}")
        print(f"Best CV accuracy: {best_score:.4f}")
        print(f"Test accuracy of best model: {test_accuracy:.4f}")

        # Log the best model to MLFlow Model Registry
        mlflow.sklearn.log_model(
            best_estimator, "model", registered_model_name=model_name_full[model_name]
        )
        print("Model logged to MLFlow Model Registry as 'LogisticRegressionModel'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model with hyperparameter tuning."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
        help="The name of the model to train.",
    )

    args = parser.parse_args()
    print("starting")
    train_model(model_name=args.model_name)

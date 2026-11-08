"""Train and evaluate five classifiers on the UCI WDBC dataset.

Outputs:
- model/*.joblib: trained model pipelines
- test_data.csv: held-out test samples including diagnosis target
- model_comparison.csv: evaluation metrics for README/reporting

The Streamlit app should load these saved models and MUST NOT retrain on test_data.csv.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = Path("wdbc.data")
MODEL_DIR = Path("model")
TEST_FILE = Path("test_data.csv")
RESULTS_FILE = Path("model_comparison.csv")

COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and validate the raw WDBC file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place wdbc.data in the same directory as this script."
        )

    df = pd.read_csv(path, header=None, names=COLUMNS)

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns but found {df.shape[1]}."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values. Please inspect wdbc.data.")

    valid_labels = {"M", "B"}
    actual_labels = set(df["diagnosis"].unique())
    if not actual_labels.issubset(valid_labels):
        raise ValueError(f"Unexpected diagnosis labels: {actual_labels}")

    # M (malignant) is the positive class, B (benign) is the negative class.
    y = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    X = df.drop(columns=["id", "diagnosis"])

    return df, X, y


def build_models() -> dict[str, tuple[object, str]]:
    """Return model display names mapped to estimator and output filename."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            "logistic_regression.joblib",
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            "decision_tree.joblib",
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]),
            "knn.joblib",
        ),
        "Naive Bayes": (
            # Scaling is saved in the pipeline so deployment preprocessing is identical.
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]),
            "naive_bayes.joblib",
        ),
        "Support Vector Machine": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]),
            "svm.joblib",
        ),
    }


def get_auc_score_values(model, X_test: pd.DataFrame):
    """Get continuous scores suitable for ROC AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise AttributeError("Model provides neither predict_proba nor decision_function.")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = get_auc_score_values(model, X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main() -> None:
    _, X, y = load_dataset(DATA_FILE)

    print("=" * 78)
    print("WDBC MACHINE LEARNING CLASSIFICATION ASSIGNMENT")
    print("=" * 78)
    print(f"Dataset samples : {len(X)}")
    print(f"Input features  : {X.shape[1]}")
    print(f"Benign (0)      : {(y == 0).sum()}")
    print(f"Malignant (1)   : {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    MODEL_DIR.mkdir(exist_ok=True)

    # Export only the unseen held-out test set. Include target as required by the assignment.
    test_df = X_test.copy()
    test_df["diagnosis"] = y_test.values
    test_df.to_csv(TEST_FILE, index=False)

    results = []

    for model_name, (model, filename) in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / filename)
        results.append({"ML Model Name": model_name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\n" + "=" * 78)
    print("MODEL EVALUATION RESULTS")
    print("=" * 78)
    display_df = results_df.copy()
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    display_df[metric_columns] = display_df[metric_columns].round(4)
    print(display_df.to_string(index=False))

    winner_index = results_df["MCC"].idxmax()
    winner = results_df.loc[winner_index, "ML Model Name"]

    print("\n" + "=" * 78)
    print(f"Models saved to       : {MODEL_DIR}/")
    print(f"Held-out test data    : {TEST_FILE}")
    print(f"Comparison metrics    : {RESULTS_FILE}")
    print(f"Highest MCC model     : {winner}")
    print("Note: determine the final overall winner using the complete metric profile,")
    print("especially AUC, Recall, F1 and MCC, not Accuracy alone.")
    print("=" * 78)


if __name__ == "__main__":
    main()

# Breast Cancer Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification models using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

All models are evaluated on the same held-out test dataset using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The project also provides a Streamlit application in which a user can upload the held-out test CSV, select one of the five trained models, and view its evaluation metrics and classification results. The Streamlit application loads pre-trained models and does not retrain models on the uploaded test data.

---

## b. Dataset Description

### Dataset

**Breast Cancer Wisconsin (Diagnostic) - WDBC**

### Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Classification Problem

This is a binary classification problem:

- `M` / `1` = Malignant
- `B` / `0` = Benign

### Dataset Characteristics

| Property | Value |
|---|---:|
| Total instances | 569 |
| Input features used | 30 |
| Original identifier | `id` |
| Target | `diagnosis` |
| Benign samples | 357 |
| Malignant samples | 212 |
| Missing feature values | None |

The input measurements are computed from digitized images of fine needle aspirates (FNA) of breast masses and describe characteristics of cell nuclei.

The dataset contains ten base measurements represented as mean, standard error and worst/largest-value measurements, producing 30 numerical input features:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

The `id` column is excluded from training because it is an identifier rather than a predictive feature. The target is encoded as `M = 1` and `B = 0`, so malignant cases are treated as the positive class when calculating Precision, Recall and F1 Score.

### Train-Test Split

A stratified 80:20 train-test split is performed using `random_state=42`.

| Split | Samples |
|---|---:|
| Training set | 455 |
| Test set | 114 |

Stratification preserves the class distribution across training and test subsets. The 114 held-out observations are exported to `test_data.csv`. They are not used to train any of the five models.

---

## c. GitHub Repository Link

**GitHub Repository:**

`<ADD YOUR GITHUB REPOSITORY LINK HERE>`

---

## Live Streamlit Application

**Streamlit App:**

`<ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

---

## d. Models Used

### 1. Logistic Regression

Logistic Regression is used as a linear probabilistic classifier. `StandardScaler` is applied through a scikit-learn pipeline so that scaling parameters are learned only from the training data and are applied consistently during testing and deployment.

### 2. Decision Tree Classifier

Decision Tree Classifier learns nonlinear decision rules by recursively dividing the feature space. Feature scaling is not required for this tree-based method.

### 3. K-Nearest Neighbors

KNN predicts the class of a sample using nearby training samples. Because KNN depends on distances between observations, `StandardScaler` is included in its saved pipeline.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes models numerical features using Gaussian distributions and applies the conditional-independence assumption to estimate class probabilities.

### 5. Support Vector Machine

SVM uses the Radial Basis Function (RBF) kernel to model nonlinear class boundaries. The model is used with standardized features and probability estimation so that probability scores can be used to calculate ROC AUC.

---

## Evaluation Metrics

| Metric | Interpretation |
|---|---|
| Accuracy | Proportion of all samples classified correctly |
| AUC | Ability of the model to rank positive cases above negative cases across thresholds |
| Precision | Proportion of predicted malignant cases that are actually malignant |
| Recall | Proportion of actual malignant cases correctly identified |
| F1 Score | Harmonic mean of Precision and Recall |
| MCC | Correlation between actual and predicted classes considering TP, TN, FP and FN |

---

## Model Comparison

All models were trained using the same training split and evaluated using the same 114-sample held-out test set.

| ML Model Name              | Accuracy   | AUC        | Precision  | Recall     | F1         |        MCC |
|----------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Logistic Regression        | 0.9649     | **0.9960** | 0.9750     | **0.9286** | 0.9512     | 0.9245     |
| Decision Tree              | 0.9298     | 0.9246     | 0.9048     | 0.9048     | 0.9048     | 0.8492     |
| KNN                        | 0.9561     | 0.9823     | 0.9744     | 0.9048     | 0.9383     | 0.9058     |
| Naive Bayes                | 0.9211     | 0.9891     | 0.9231     | 0.8571     | 0.8889     | 0.8292     |
| **Support Vector Machine** | **0.9737** | 0.9947     | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

---

## Performance Observations

| ML Model Name          | Observation About Model Performance |
|----------------------  |-------------------------------------|
| Logistic Regression    | Logistic Regression performed strongly with 96.49% accuracy and achieved the highest AUC of 0.9960. Its recall of 92.86% indicates strong detection of malignant samples, while an MCC of 0.9245 indicates excellent agreement between actual and predicted classes. |
| Decision Tree          | Decision Tree achieved 92.98% accuracy, an AUC of 0.9246 and an MCC of 0.8492. Although its performance was good, it was weaker than Logistic Regression, KNN and SVM across several important evaluation metrics.                                                      |
| KNN                    | KNN achieved 95.61% accuracy, an AUC of 0.9823 and an MCC of 0.9058. Precision was high at 97.44%, while recall was 90.48%. Feature standardization is particularly important for KNN because its predictions depend on distances between observations.                  |
| Naive Bayes            | Naive Bayes achieved 92.11% accuracy and a strong AUC of 0.9891. However, it had the lowest recall at 85.71%, the lowest F1 Score at 0.8889 and the lowest MCC at 0.8292 among the five evaluated models.                                                                     |
| Support Vector Machine | SVM produced the strongest overall performance. It achieved the highest accuracy of 97.37%, perfect precision of 100% on the held-out test set, a recall of 92.86%, the highest F1 Score of 0.9630 and the highest MCC of 0.9442. Its AUC was also excellent at 0.9947. |
| **Overall Winner** | **Support Vector Machine (SVM)** is selected as the overall winner because it achieved the highest Accuracy, Precision, F1 Score and MCC while tying for the highest Recall and maintaining an excellent AUC. |

### Overall Result

**Support Vector Machine is the best-performing model for this experiment.**

Logistic Regression achieved a slightly higher AUC (`0.9960`) than SVM (`0.9947`). However, SVM achieved better Accuracy, Precision, F1 Score and MCC while matching Logistic Regression's Recall. Therefore, SVM provides the strongest overall metric profile rather than being selected using Accuracy alone.

Because malignant diagnosis is encoded as the positive class, Recall is particularly important in this experiment. SVM and Logistic Regression both achieved a malignant-class Recall of `0.9286`, meaning 92.86% of malignant samples in this held-out test set were correctly identified.

The SVM Precision of `1.0000` means that every sample predicted as malignant by SVM in this specific held-out test set was actually malignant. This result applies to this experiment and test split and should not be interpreted as a general clinical performance guarantee.

---

## Repository Structure

```text
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── wdbc.data
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## Requirements

The project uses the following Python packages:

```text
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Training

Ensure `wdbc.data` is located in the project root and run:

```bash
python3 train_models.py
```

The training script performs the following operations:

1. Loads the original `wdbc.data` file.
2. Assigns the WDBC feature names.
3. Removes the `id` field from model inputs.
4. Encodes `B = 0` and `M = 1`.
5. Creates a stratified 80:20 train-test split.
6. Trains all five required classifiers.
7. Calculates Accuracy, AUC, Precision, Recall, F1 and MCC.
8. Saves all trained models/pipelines in the `model/` directory.
9. Exports the untouched held-out test subset as `test_data.csv`.
10. Exports the comparison results as `model_comparison.csv`.

---

## Saved Models

After training, the following model artifacts are produced:

```text
model/logistic_regression.joblib
model/decision_tree.joblib
model/knn.joblib
model/naive_bayes.joblib
model/svm.joblib
```

The preprocessing required for scale-sensitive algorithms is stored with the relevant model using scikit-learn pipelines. This ensures that the same preprocessing used during model training is automatically applied during Streamlit evaluation.

---

## Test Data

`test_data.csv` contains the held-out test subset generated by `train_models.py`.

It contains:

- All 30 required input features
- The `diagnosis` target column

This file is intended for evaluating the saved models through the Streamlit application. It must not be used to retrain the models.

---

## Run the Streamlit Application Locally

After installing dependencies, start the application with:

```bash
streamlit run app.py
```

Then:

1. Upload `test_data.csv`.
2. Select one of the five classification models.
3. View the evaluation metrics.
4. View the confusion matrix and classification report.

---

## Streamlit Application Features

The Streamlit application provides the assignment-required functionality:

- Dataset upload option for CSV test data
- Model selection dropdown
- Loading of saved pre-trained models
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report
- Evaluation without retraining on uploaded test data

---

## Streamlit Community Cloud Deployment

1. Push all required project files to GitHub.
2. Go to Streamlit Community Cloud.
3. Sign in using the GitHub account containing this repository.
4. Select **New App**.
5. Select this repository.
6. Select the `main` branch.
7. Set the application file to `app.py`.
8. Deploy the application.
9. Upload `test_data.csv` and verify the results for all five classifiers.

---

## BITS Virtual Lab Execution

The machine learning experiment was executed on BITS Virtual Lab using:

```bash
python3 train_models.py
```

The execution produced results for all five classifiers on the same held-out test set and generated the saved model artifacts and test CSV.

A single screenshot of the BITS Virtual Lab execution showing the model evaluation results should be included in the final assignment PDF as execution proof.

---

## Reproducibility and Prevention of Data Leakage

The implementation uses the following controls to ensure a fair comparison:

- All five classifiers use the same training and test split.
- `random_state=42` is used where applicable for reproducibility.
- A stratified split preserves class distribution.
- Test samples are excluded from model training.
- Scaling is fitted using training data through scikit-learn pipelines.
- Saved pipelines retain the required preprocessing.
- `test_data.csv` contains only the held-out test subset.
- The Streamlit application evaluates saved models and does not call `fit()` on uploaded test data.

---

## Conclusion

Five machine learning classification algorithms were evaluated on the Breast Cancer Wisconsin (Diagnostic) dataset using six evaluation metrics. All models were trained using the same training subset and evaluated using the same held-out test data.

SVM achieved the strongest overall performance, with an Accuracy of `0.9737`, Precision of `1.0000`, Recall of `0.9286`, F1 Score of `0.9630`, MCC of `0.9442`, and AUC of `0.9947`. Logistic Regression achieved the highest individual AUC of `0.9960`, but SVM performed better across most of the other key metrics. Therefore, **Support Vector Machine was selected as the overall winning model for this dataset and experimental split**.

---

## Final Submission Checklist

Before generating the final PDF, verify that:

- [x] Dataset satisfies at least 500 instances and 12 features
- [x] All five required classifiers have been trained
- [x] All six required metrics have been calculated
- [x] `test_data.csv` has been generated
- [x] Five trained models have been saved in `model/`
- [x] `requirements.txt` has been created
- [x] README contains the model comparison and observations
- [ ] `app.py` is complete and tested
- [ ] GitHub repository link has been inserted above
- [ ] Streamlit Community Cloud link has been inserted above
- [ ] BITS Virtual Lab screenshot has been added to the final PDF
- [ ] README content has been included in the final PDF

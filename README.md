
# BankWise — Bank Marketing Response Prediction


## Live Demo

Try the deployed application: [BankWise — Live App](PASTE_YOUR_STREAMLIT_APP_URL_HERE)


A machine learning project that predicts whether a customer is likely to subscribe to a bank term deposit based on customer information and marketing campaign details.

The project compares four classification models and provides an interactive Streamlit application for testing predictions.

## Project Objectives

- Explore and preprocess the Bank Marketing dataset.
- Train and compare four classification models.
- Evaluate model performance using accuracy, precision, recall and F1-score.
- Provide an interactive interface for entering customer details and viewing predictions.

## Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing dataset.

The `bank-full.csv` dataset contains 45,211 records and 17 columns, including customer characteristics, campaign information and the subscription outcome.

- **Input features:** Customer and campaign attributes.
- **Target:** `y`, indicating whether the customer subscribed (`yes` or `no`).
- **Excluded feature:** `duration`, because call duration is known only after a call and would not be available when making a pre-call prediction.

## Data Preprocessing

1. Loaded the CSV using Pandas with `;` as the separator.
2. Inspected the data, target distribution, unknown values and duplicate records.
3. Separated the input features and target.
4. Applied one-hot encoding to categorical features using `pd.get_dummies()` with `drop_first=True`.
5. Split the data into 80% training and 20% testing sets using stratified sampling and `random_state=42`.
6. Applied `StandardScaler`, fitting it on the training data and transforming both training and testing data.
7. Saved the fitted scaler and encoded feature-column list for use in the application.

## Machine Learning Models

The project compares:

1. **Balanced Logistic Regression** — Logistic Regression with `class_weight="balanced"`.
2. **Decision Tree** — Decision Tree classifier with balanced class weights.
3. **Random Forest** — Random Forest classifier with balanced class weights.
4. **Balanced Random Forest** — An ensemble model designed to address class imbalance through balanced sampling.

## Model Evaluation

The following results were recorded in the notebook. Precision, recall and F1-score are calculated for the `yes` class.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Balanced Logistic Regression | 75.57% | 26.69% | 62.29% | 37.37% |
| Decision Tree | 84.35% | 32.02% | 30.06% | 31.01% |
| Random Forest | 88.00% | 48.43% | 39.32% | 43.40% |
| Balanced Random Forest | 84.16% | 38.37% | 58.32% | 46.29% |

### Key Findings

- **Balanced Random Forest** achieved the highest F1-score (46.29%) among the four tested models.
- **Random Forest** achieved the highest accuracy (88.00%) and precision (48.43%).
- **Balanced Logistic Regression** achieved the highest recall (62.29%).

These results demonstrate a trade-off between identifying potential subscribers and limiting false-positive predictions. The best model depends on the intended business objective.

## Interactive Application

The Streamlit application allows users to enter customer details and obtain predictions from all four saved models.

The interface includes:
- Customer input fields.
- Predictions from all four classifiers.
- A model-performance comparison table.

Predictions are estimates and should not be treated as guarantees of actual customer behaviour.

## Project Structure

```text
BankMarketing/
├── app.py
├── bank-full.csv
├── balanced_logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── balanced_random_forest.pkl
├── scaler.pkl
├── model_columns.pkl
├── README.md
└── .streamlit/
    └── config.toml
```

The notebook used for data exploration, training and evaluation can also be included in the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd BankMarketing
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment (optional but recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install streamlit pandas scikit-learn imbalanced-learn joblib
```

Use compatible versions of the libraries used to train the saved models.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, typically `http://localhost:8501`.

## Limitations

- The dataset has an imbalanced target distribution.
- Predictions depend on the training data and selected features.
- Accuracy alone is not sufficient to evaluate performance on an imbalanced dataset.
- The current interface is a demonstration and has not been established as a production banking system.
- The three manual customer tests are functional checks, not a measure of general model accuracy.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit

## Disclaimer

This project is for educational and demonstration purposes only. It is not intended to make real banking or financial decisions.

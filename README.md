# titanic-survival-prediction
This project uses machine learning to predict passenger survival on the Titanic, based on Kaggle's Titanic dataset.

## 🔍 Problem Statement

The Titanic dataset is a classic binary classification problem. Given features like passenger class, sex, age, and fare, the goal is to predict whether a passenger survived the disaster.

## 📁 Project Structure
titanic-survival-prediction/
├── titanic.ipynb # Main notebook
├── submission.csv # Final predictions for Kaggle
├── data/
│ ├── train.csv
│ └── test.csv
└── README.md

## 🧠 Model Pipeline

1. **Data Cleaning**: 
   - Handle missing values (Age, Embarked).
   - Drop irrelevant columns (Cabin, Ticket).
   - Log transform on Fare.
   - Create features like `FamilySize`, `IsAlone`, and extract `Title` from name.

2. **Encoding**:
   - One-hot encoding on categorical variables (`Sex`, `Embarked`, `Title`).

3. **Model Training**:
   - Compared 3 models: Logistic Regression, Random Forest, and K-Nearest Neighbors.
   - Used `GridSearchCV` to tune hyperparameters.
   - Final model: `Random Forest Classifier`.

4. **Evaluation**:
   - Evaluated using Accuracy, Precision, Recall, F1 Score, and ROC AUC.
   - Achieved **~78% Kaggle public score**.

## 📈 Performance

| Metric     | Score (Validation) |
|------------|--------------------|
| Accuracy   | ~0.81              |
| Precision  | ~0.78              |
| Recall     | ~0.80              |
| F1 Score   | ~0.79              |
| Kaggle Public Score | **0.78708** |

## 💾 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
requirements.txt:
pandas
numpy
scikit-learn
matplotlib
seaborn

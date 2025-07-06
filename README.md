# Credit Card Fraud Detection using XGBoost 
This project builds and optimizes a machine learning model to detect fraudulent credit card transactions. It handles class imbalance using SMOTE and evaluates performance using ROC AUC. It also uses cross-validation and hyperparameter tuning via RandomizedSearchCV.

## Project Overview
Dataset: Anonymized credit card transaction data with imbalanced labels.

Model: XGBoost Classifier.

Imbalance Handling: SMOTE (Synthetic Minority Oversampling Technique).

Evaluation: ROC AUC.

Optimization: Cross-validation and RandomizedSearchCV with a pipeline.

## Dataset Details
The dataset is a .csv file with the following characteristics:

Feature columns: PCA-anonymized features V1, V2, ..., V28 + Time, Amount.

Target column: Class (0 = normal, 1 = fraud).

## Running the Program

Clone or download the respository, then open a terminal inside the project directory and instlal the dependencies for the project:

pip install -r requirements.txt

Then run: 

python fraud_detection.py

This will:

Load and inspect the dataset.

Apply SMOTE to rebalance the training data.

Train an XGBoost model and evaluate it.

Perform cross-validation to validate robustness.

Use RandomizedSearchCV with a pipeline to find the best hyperparameters.



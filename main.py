import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline


data = pd.read_csv('creditcard.csv')

print(data.head())
print("Shape:", data.shape)
data["Class"].unique()

# Model Training

y = data['Class']
X = data.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use SMOTE to oversample fraudulent transactions
counter =  Counter(y_train)
print("Before SMOTE:", counter)
smote = SMOTE(random_state=0)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_sm))

# Train the model
model = XGBClassifier(eval_metric='auc',random_state=0)
model.fit(X_train_sm, y_train_sm)

predictions = model.predict(X_test)

# Evaluate  using accuracy and ROC AUC
accuracy = accuracy_score(y_test, predictions)
auc_score = roc_auc_score(y_test, predictions)
print("Base model Accuracy:", accuracy)
print("Base model ROC AUC Score:", auc_score)

conf = confusion_matrix(y_test, predictions)
print("Total Fraud:", y_test.sum())
print("Total True Negatives:", conf[0][0])
print("Total True Positives:", conf[1][1])
print("Total False positives:", conf[0][1])
print("Total False negatives:", conf[1][0])

# Model Optimization

# Apply cross-validation
k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
scores = []

for train_index, test_index in k_folds.split(X):
    X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
    y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]

    X_resampled, y_resampled = smote.fit_resample(X_fold_train, y_fold_train)

    model.fit(X_resampled, y_resampled)
    predictions = model.predict(X_fold_test)

    score = roc_auc_score(y_fold_test, predictions)
    scores.append(score)
print("Average ROC AUC Score using cross validation:", np.mean(scores))

# Tune Hyperparameters

# Make a pipeline to apply SMOTE in RandomizedSearchCV
pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=0)),
    ('model', XGBClassifier(eval_metric='auc', random_state=0)),
])

params = {'model__n_estimators': [200, 300], 'model__max_depth': [6, 8, 10]}

search = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=3, scoring='roc_auc', cv=5, random_state=0, n_jobs=-1)
search.fit(X_train, y_train)

print("Best Parameters using RandomizedSearchCV:", search.best_params_)
print("Best AUC Score:", search.best_score_)

best_model = search.best_estimator_

new_predictions = best_model.predict(X_test)
new_auc = roc_auc_score(y_test, new_predictions)
print("Test AUC with RandomizedSearchCV:", new_auc)
new_conf = confusion_matrix(y_test, new_predictions)
print("Scores After Hyperparameter Tuning:")
print("Total True Negative:", new_conf[0][0])
print("Total True Positives:", new_conf[1][1])
print("Total False positives:", new_conf[0][1])
print("Total False negatives:", new_conf[1][0])

import pandas as pd
import numpy as np
import xgboost as xgb
import ast
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score


# Load data
legit_df = pd.read_csv("LegitFeatures.csv")
ponzi_df = pd.read_csv("PonziFeatures.csv")

# Create target variable
legit_df['target'] = 0
ponzi_df['target'] = 1

# Concatenate data
data = pd.concat([legit_df, ponzi_df], ignore_index=True)

# Convert the 'instruction_counts' string into a dictionary
data['instruction_counts'] = data['instruction_counts'].apply(ast.literal_eval)

# Convert the 'instruction_counts' dictionary into a DataFrame with dtype 'float64'
instruction_counts_df = data['instruction_counts'].apply(lambda x: pd.Series(x, dtype='float64'))

# Fill NaN values with 0
instruction_counts_df = instruction_counts_df.fillna(0)

# Concatenate the instruction counts DataFrame with the original DataFrame, including the 'target' variable
data = pd.concat([data.drop(columns=['instruction_counts']), instruction_counts_df], axis=1)
# Drop opcodes and balance columns
data = data.drop(columns=['address', 'opcodes', 'bytecode', 'balance', 'addr_get_profit'])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Export the combined dataset to a new CSV file
data.to_csv('CombineFeatures.csv', index=False)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['target']), data['target'], test_size=0.3, stratify=data['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

# Convert column names to strings
X_train.columns = X_train.columns.astype(str)

# Resample the dataset
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Rename columns in X_train to remove special characters
X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)

# Rename columns in X_val to remove special characters
X_val.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_val.columns]

# Create DMatrix for XGBoost
dval = xgb.DMatrix(X_val, label=y_val)

# Rename columns in X_test to remove special characters
X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_test.columns]

# Create DMatrix for XGBoost
dtest = xgb.DMatrix(X_test, label=y_test)

# Feature selection using mutual information
k = 10  # Number of top features to select
selector = SelectKBest(mutual_info_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_selected)
X_val_pca = pca.transform(X_val_selected)
X_test_pca = pca.transform(X_test_selected)

# Early stopping
eval_metric = 'auc'

# Stacking ensemble
base_models = [
    ('xgb', xgb.XGBClassifier(objective='binary:logistic')),
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier()),
    ('lgb', lgb.LGBMClassifier())
]

stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
stacking_model.fit(X_train_pca, y_train)

# Evaluate and plot the results
y_pred = stacking_model.predict(X_test_pca)
y_prob = stacking_model.predict_proba(X_test_pca)[:, 1]
print('Test set performance:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Compute and plot ROC curve
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()


# Plot the feature importances for the XGBoost base model
xgb_model = stacking_model.named_estimators_['xgb']
# Fit the XGBoost base model on the selected features
xgb_model.fit(X_train_selected, y_train, eval_set=[(X_val_selected, y_val)], early_stopping_rounds=10, verbose=False)
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title('Feature Importances (XGBoost Base Model)')
plt.bar(range(X_train_selected.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X_train_selected.shape[1]), X_train.columns[selector.get_support()][indices], rotation=90)
plt.xlim([-1, X_train_selected.shape[1]])
plt.tight_layout()
plt.show()

# Calculate permutation importances for the XGBoost base model
result_base = permutation_importance(xgb_model, X_test_selected, y_test, n_repeats=10, random_state=0, n_jobs=-1)

# Get importances and their indices for the XGBoost base model
importances_base = result_base.importances_mean
indices_base = np.argsort(importances_base)[::-1]

# Plot the permutation importances for the XGBoost base model
plt.figure()
plt.title("Permutation Importances (XGBoost Base Model)")
plt.bar(range(X_test_selected.shape[1]), importances_base[indices_base], color="b", align="center")
plt.xticks(range(X_test_selected.shape[1]), X_train.columns[selector.get_support()][indices_base], rotation=90)
plt.xlim([-1, X_test_selected.shape[1]])
plt.tight_layout()
plt.show()

# Bayesian optimization
param_space = {
    'xgb__max_depth': Integer(1, 7),
    'xgb__min_child_weight': Integer(1, 7),
    'xgb__subsample': Real(0.1, 0.9),
    'xgb__colsample_bytree': Real(0.1, 0.9),
    'xgb__gamma': Real(0, 7),
    'xgb__alpha': Real(0, 7),
    'xgb__lambda': Real(0, 7),
    'xgb__booster': Categorical(['gbtree', 'gblinear', 'dart']),
    'xgb__scale_pos_weight': Real(1, 7)
}

# Create a dictionary of multiple scoring metrics
scoring_metrics = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

cv = StratifiedKFold(n_splits=5, shuffle=True)

bayes_search = BayesSearchCV(
    estimator=stacking_model,
    search_spaces=param_space,
    scoring=scoring_metrics,
    n_iter=200,
    cv=cv,
    n_jobs=12,
    verbose=2,
    refit="recall",  # Specify the metric to use for refitting the best model
)

bayes_search.fit(X_train_selected, y_train)

# Print best hyperparameters and best F1 score from Bayesian optimization
print('Best hyperparameters: ', bayes_search.best_params_)
print('Best recall score: {:.4f}'.format(bayes_search.best_score_))

# Obtain the best model from the Bayesian optimization process
best_model = bayes_search.best_estimator_

# Predict the probabilities for the test set using the best model
y_prob_best_model = best_model.predict_proba(X_test_selected)[:, 1]

# Calculate the AUC score for the best model
roc_auc_best_model = roc_auc_score(y_test, y_prob_best_model)

# Evaluate and plot the results
y_pred_best_model = best_model.predict(X_test_selected)
y_prob_best_model = best_model.predict_proba(X_test_selected)[:, 1]

# Evaluate performance on test set
print('Test set performance (Best Model):')
print(classification_report(y_test, y_pred_best_model))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_best_model))

# Plot the ROC curve for the best model
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_prob_best_model)
plt.figure()
plt.plot(fpr_best, tpr_best, label='ROC curve (AUC = {:.2f})'.format(roc_auc_best_model))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve (Stack Model)')
plt.legend(loc="lower right")
plt.show()

# Feature importances for the XGBoost model
xgb_best_model = best_model.named_estimators_['xgb']
importances_best = xgb_best_model.feature_importances_
indices_best = np.argsort(importances_best)[::-1]
plt.figure()
plt.title('Feature Importances (Best Model)')
plt.bar(range(X_train_selected.shape[1]), importances_best[indices_best], color='b', align='center')
plt.xticks(range(X_train_selected.shape[1]), X_train.columns[selector.get_support()][indices_best], rotation=90)
plt.xlim([-1, X_train_selected.shape[1]])
plt.tight_layout()
plt.show()

# Calculate permutation importances for the best model
result_best_model = permutation_importance(best_model, X_test_selected, y_test, n_repeats=10, random_state=0, n_jobs=-1)

# Get importances and their indices for the best model
importances_best_model = result_best_model.importances_mean
indices_best_model = np.argsort(importances_best_model)[::-1]

# Plot the permutation importances for the best model
plt.figure()
plt.title("Permutation Importances (Best Model)")
plt.bar(range(X_test_selected.shape[1]), importances_best_model[indices_best_model], color="b", align="center")
plt.xticks(range(X_test_selected.shape[1]), X_train.columns[selector.get_support()][indices_best_model], rotation=90)
plt.xlim([-1, X_test_selected.shape[1]])
plt.tight_layout()
plt.show()
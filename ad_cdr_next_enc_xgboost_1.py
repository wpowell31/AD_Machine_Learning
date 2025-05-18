import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import shap
import warnings
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
print("imports complete")

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/cdr_next_encounter/cdr_next_enc_clinical_additional.csv')
#df = df[df['DAYS_TO_NEXT_CDR_SCORE'] >= 180]
df = df[df['VALUE'].isin([0.0, 0.5])]

# Separate features and the label
X = df.drop(columns=['EPIC_MRN', 'PT_LAST_NAME', 'PT_FIRST_NAME', 'DAYS_SINCE_LAST_CDR_TEST', 'NEXT_CDR_VALUE', 'PERFORMED',
       'PT_MIDDLE_NAME', 'PT_BIRTH_DTTM', 'PT_DEATH_DTTM', 'PT_GENDER',
       'PT_ETHNIC_GROUP', 'ADDRESS_LINE', 'PT_ADDRESS', 'PT_CITY', 'PT_STATE', 'CDR_ID', 'CDR_NUMBER', 'DAYS_TO_NEXT_CDR_SCORE',
       'PT_ZIPCODE'])
y = (df['NEXT_CDR_VALUE'] >= 1.0).astype(int)

# Calculate class balance
class_balance = np.bincount(y) / len(y)
positive_weight = class_balance[0] / class_balance[1]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
X_test.columns = X_train.columns

print(positive_weight)
# Define the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 4, 5, 6, 7],
    #'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the model on the test set
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Metrics
auroc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/cdr_next_encounter/additional_covariates/xgb_roc_curve.png')

# Save metrics to a text file
with open('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/cdr_next_encounter/additional_covariates/xgb_model_metrics_greater_than_1.txt', 'w') as f:
    f.write(f"Class Balance:\n{class_balance}\n")
    f.write(f"Best Hyperparameters:\n{best_params}\n")
    f.write(f"AUROC: {auroc}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# Generate SHAP summary plot
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/cdr_next_encounter/additional_covariates/xgb_shap_summary_plot.png', bbox_inches='tight')
plt.close()



print("Metrics saved to 'xgb_model_metrics.txt' and SHAP summary plot saved to 'xgb_shap_summary_plot.png'")
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/df_clinical_combined_temp.csv')
# Separate features and the label
X = df.drop(columns=['EPIC_MRN', 'PERFORMED', 'VALUE', 'PT_BIRTH_DTTM', 'Dementias', 'Alzheimer\'s disease', 'VALUE_GS_1'])
y = df['VALUE_GS_1']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
X_test.columns = X_train.columns

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='logloss')

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 5],
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
plt.savefig('roc_curve.png')
plt.close()


# Save metrics to a text file
with open('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/xgb_model_metrics.txt', 'w') as f:
    f.write(f"Best Hyperparameters:\n{best_params}\n")
    f.write(f"AUROC: {auroc}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# rename columns that are too long
X_train = X_train.rename(
    columns={
        'Persons with potential health hazards related to socioeconomic, psychosocial, and other circumstances': 'socioeconomic health hazards',
        'Symptoms concerning nutrition, metabolism, and development': 'nutrition/metabolism symptoms',
        'Symptoms involving nervous and musculoskeletal systems': 'nervous/musculoskeletal symptoms'
    }
)
# Generate SHAP summary plot
plt.figure()
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/xgb_shap_summary_plot.png', bbox_inches='tight')
plt.close()


print("Metrics saved to 'model_metrics.txt' and SHAP summary plot saved to 'shap_summary_plot.png'")
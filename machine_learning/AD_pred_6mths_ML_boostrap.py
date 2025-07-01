import catboost as cb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
import shap
import warnings
import matplotlib.pyplot as plt

print("Imports Complete")
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/6months/df_extract_ad_prediction_6mths2.csv')

# Function to parse the 'PT_BIRTH_DTTM' column based on format
def parse_first_enc(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        return pd.to_datetime(date_str, format='%Y-%m-%d')

# Apply the function to the 'FIRST_ENC' column
df['FIRST_ENC'] = df['FIRST_ENC'].apply(parse_first_enc)

# Convert other date columns
date_columns_with_time = ['LAST_ENC']
date_columns_without_time = ['PT_BIRTH_DTTM', 'INDEX_DATE']

for col in date_columns_with_time:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

for col in date_columns_without_time:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

df = df[df['INDEX_DATE'] >= df['FIRST_ENC']]


# Separate features and the label
X = df.drop(columns=['EPIC_MRN', 'FIRST_ENC', 'LAST_ENC', 'DIFF_YEARS', 'INDEX_TIME_ENC',
       'FIRST_AD_CDR_DATE', 'FIRST_AD_FLOWSHEET_DATE',
       'FIRST_AD_DIAGNOSIS_DATE', 'FIRST_AD_MED_DATE', 'EARLIEST_OVERALL_DATE',
       'INDEX_DATE', 'AD_label', 'PT_LAST_NAME', 'PT_FIRST_NAME',
       'PT_MIDDLE_NAME', 'PT_BIRTH_DTTM', 'PT_DEATH_DTTM', 
       'PT_ETHNIC_GROUP', 'ADDRESS_LINE', 'PT_ADDRESS', 'PT_CITY', 'PT_STATE',
       'PT_ZIPCODE'])
y = df['AD_label']

# drop na
#X = X.dropna()
X = X.fillna(0)
print(f"X.shape: {X.shape}")
# Drop columns where every entry is 0
X = X.loc[:, (X != 0).any(axis=0)]
print(f"X.shape: {X.shape}")
# Calculate the proportion of 0s in each column
zero_proportion = (X == 0).mean()

# Drop columns where the proportion of 0s is greater than 95%
X = X.loc[:, zero_proportion <= 0.99]
print(f"X.shape: {X.shape}")

X_indices = X.index

# Drop corresponding indices from y
y = y.loc[X_indices]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
X_test.columns = X_train.columns

# Define the models and hyperparameter grids
models = {
    'CatBoost': cb.CatBoostClassifier(eval_metric='Logloss', silent=True),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

param_grids = {
    'CatBoost': {
        'iterations': [50, 100, 150],
        'depth': [4, 5],
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# Dictionary to store the best models
best_models = {}

# Loop through each model for hyperparameter tuning
for model_name in models:
    print(f"Performing GridSearchCV for {model_name}")
    model = models[model_name]
    param_grid = param_grids[model_name]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=10, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    best_models[model_name] = best_model
    print(f"Best Hyperparameters for {model_name}: {best_params}")
    
    # Monte Carlo Cross-Validation
    n_iterations = 100
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    
    # Perform bootstrap resampling for calculating AUROC CI
    bootstrapped_auroc = []
    for i in range(n_iterations):
        #X_train_resampled, y_train_resampled = resample(X_train, y_train, random_state=i)
        X_test_resampled, y_test_resampled = resample(X_test, y_test, random_state=i)

        #best_model.fit(X_train_resampled, y_train_resampled)
        
        y_test_pred_prob = best_model.predict_proba(X_test_resampled)[:, 1]
        y_test_pred = best_model.predict(X_test_resampled)
        
        metrics['accuracy'].append(accuracy_score(y_test_resampled, y_test_pred))
        metrics['precision'].append(precision_score(y_test_resampled, y_test_pred))
        metrics['recall'].append(recall_score(y_test_resampled, y_test_pred))
        metrics['f1'].append(f1_score(y_test_resampled, y_test_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test_resampled, y_test_pred_prob))

        bootstrapped_auroc.append(roc_auc_score(y_test_resampled, y_test_pred_prob))

    metric_summary = {}
    for metric in metrics:
        mean_metric = np.mean(metrics[metric])
        lower_bound = np.percentile(metrics[metric], 2.5)
        upper_bound = np.percentile(metrics[metric], 97.5)
        metric_summary[metric] = (mean_metric, lower_bound, upper_bound)

    print(f"Metrics with 95% confidence intervals for {model_name}:")
    for metric in metric_summary:
        print(f"{metric.capitalize()}: Mean = {metric_summary[metric][0]:.4f}, 95% CI = [{metric_summary[metric][1]:.4f}, {metric_summary[metric][2]:.4f}]")

    auroc_mean = np.mean(bootstrapped_auroc)
    auroc_ci_lower = np.percentile(bootstrapped_auroc, 2.5)
    auroc_ci_upper = np.percentile(bootstrapped_auroc, 97.5)
    print(f"AUROC for {model_name}: Mean = {auroc_mean:.4f}, 95% CI = [{auroc_ci_lower:.4f}, {auroc_ci_upper:.4f}]")
    
    with open(f'/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/6months/{model_name.lower()}_model_metrics_bootstrap.txt', 'a') as f:
        f.write("Hyperparameter Tuning Boostrap Resampling Metrics:\n")
        for metric in metric_summary:
            f.write(f"{metric.capitalize()}: Mean = {metric_summary[metric][0]:.4f}, 95% CI = [{metric_summary[metric][1]:.4f}, {metric_summary[metric][2]:.4f}]\n")
        f.write(f"AUROC: Mean = {auroc_mean:.4f}, 95% CI = [{auroc_ci_lower:.4f}, {auroc_ci_upper:.4f}]\n")
        print(f"Hyperparameter Tuning Boostrap Resampling metrics saved to '{model_name.lower()}_model_metrics_bootstrap.txt'")

    # Generate SHAP summary plot
    print(f"Generating SHAP summary plot for {model_name}")
    plt.figure()
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f'/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/6months/{model_name.lower()}_shap_summary_plot_bootstrap.png', bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to '{model_name.lower()}_shap_summary_plot_bootstrap.png'")

print("All models processed successfully.")
